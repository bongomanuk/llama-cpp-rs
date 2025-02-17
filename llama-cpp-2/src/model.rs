//! A safe wrapper around `llama_model`.
use std::ffi::CString;
use std::num::NonZeroU16;
use std::os::raw::c_int;  // Only keep c_int since c_char isn't used directly
use std::path::Path;
use std::ptr::NonNull;

// Only use u8 for aarch64-linux
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
type CChar = u8;

#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
type CChar = std::os::raw::c_char;  // Use full path instead of import

use crate::context::params::LlamaContextParams;
use crate::context::LlamaContext;
use crate::llama_backend::LlamaBackend;
use crate::model::params::LlamaModelParams;
use crate::token::LlamaToken;
use crate::token_type::{LlamaTokenAttr, LlamaTokenAttrs};
use crate::{
    ApplyChatTemplateError, ChatTemplateError, LlamaContextLoadError, LlamaLoraAdapterInitError,
    LlamaModelLoadError, NewLlamaChatMessageError, StringToTokenError, TokenToStringError,
};

pub mod params;

/// A safe wrapper around `llama_model`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaModel {
    pub(crate) model: NonNull<llama_cpp_sys_2::llama_model>,
}

/// A safe wrapper around `llama_lora_adapter`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaLoraAdapter {
    pub(crate) lora_adapter: NonNull<llama_cpp_sys_2::llama_adapter_lora>,
}

/// A Safe wrapper around `llama_chat_message`
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LlamaChatMessage {
    role: CString,
    content: CString,
}

impl LlamaChatMessage {
    /// Create a new `LlamaChatMessage`
    ///
    /// # Errors
    /// If either of ``role`` or ``content`` contain null bytes.
    pub fn new(role: String, content: String) -> Result<Self, NewLlamaChatMessageError> {
        Ok(Self {
            role: CString::new(role)?,
            content: CString::new(content)?,
        })
    }
}

/// How to determine if we should prepend a bos token to tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddBos {
    /// Add the beginning of stream token to the start of the string.
    Always,
    /// Do not add the beginning of stream token to the start of the string.
    Never,
}

/// How to determine if we should tokenize special tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Special {
    /// Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext. Does not insert a leading space.
    Tokenize,
    /// Treat special and/or control tokens as plaintext.
    Plaintext,
}

unsafe impl Send for LlamaModel {}

unsafe impl Sync for LlamaModel {}

impl LlamaModel {
    pub(crate) fn vocab_ptr(&self) -> *const llama_cpp_sys_2::llama_vocab {
        unsafe { llama_cpp_sys_2::llama_model_get_vocab(self.model.as_ptr()) }
    }

    /// get the number of tokens the model was trained on
    ///
    /// # Panics
    ///
    /// If the number of tokens the model was trained on does not fit into an `u32`. This should be impossible on most
    /// platforms due to llama.cpp returning a `c_int` (i32 on most platforms) which is almost certainly positive.
    #[must_use]
    pub fn n_ctx_train(&self) -> u32 {
        let n_ctx_train = unsafe { llama_cpp_sys_2::llama_n_ctx_train(self.model.as_ptr()) };
        u32::try_from(n_ctx_train).expect("n_ctx_train fits into an u32")
    }

    /// Get all tokens in the model.
    pub fn tokens(
        &self,
        special: Special,
    ) -> impl Iterator<Item = (LlamaToken, Result<String, TokenToStringError>)> + '_ {
        (0..self.n_vocab())
            .map(LlamaToken::new)
            .map(move |llama_token| (llama_token, self.token_to_str(llama_token, special)))
    }

    /// Get the beginning of stream token.
    #[must_use]
    pub fn token_bos(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_token_bos(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the end of stream token.
    #[must_use]
    pub fn token_eos(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_token_eos(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the newline token.
    #[must_use]
    pub fn token_nl(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_token_nl(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Check if a token represents the end of generation (end of turn, end of sequence, etc.)
    #[must_use]
    pub fn is_eog_token(&self, token: LlamaToken) -> bool {
        // Also check our last token to avoid repeats
        static mut LAST_TOKEN: Option<LlamaToken> = None;
        unsafe {
            let is_repeat = LAST_TOKEN == Some(token);
            LAST_TOKEN = Some(token);
            is_repeat || token == self.token_eos() || llama_cpp_sys_2::llama_token_is_eog(self.vocab_ptr(), token.0)
        }
    }

    /// Get the decoder start token.
    #[must_use]
    pub fn decode_start_token(&self) -> LlamaToken {
        let token =
            unsafe { llama_cpp_sys_2::llama_model_decoder_start_token(self.model.as_ptr()) };
        LlamaToken(token)
    }

    /// Convert single token to a string.
    ///
    /// # Errors
    ///
    /// See [`TokenToStringError`] for more information.
    pub fn token_to_str(
        &self,
        token: LlamaToken,
        special: Special,
    ) -> Result<String, TokenToStringError> {
        let bytes = self.token_to_bytes(token, special)?;
        Ok(String::from_utf8(bytes)?)
    }

    /// Convert single token to bytes.
    ///
    /// # Errors
    /// See [`TokenToStringError`] for more information.
    ///
    /// # Panics
    /// If a [`TokenToStringError::InsufficientBufferSpace`] error returned by
    /// [`Self::token_to_bytes_with_size`] contains a positive nonzero value. This should never
    /// happen.
    pub fn token_to_bytes(
        &self,
        token: LlamaToken,
        special: Special,
    ) -> Result<Vec<u8>, TokenToStringError> {
        // Only filter true EOS tokens
        if token == self.token_eos() && self.is_eog_token(token) {
            return Ok(b"\n".to_vec()); // Convert EOS to newline 
        }
        
        match self.token_to_bytes_with_size(token, 8, special, None) {
            Err(TokenToStringError::InsufficientBufferSpace(i)) => self.token_to_bytes_with_size(
                token,
                (-i).try_into().expect("Error buffer size is positive"),
                special,
                None,
            ),
            x => x,
        }
    }
}
