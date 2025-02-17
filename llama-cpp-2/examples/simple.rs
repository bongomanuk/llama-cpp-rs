//! This is a translation of simple.cpp in llama.cpp using llama-cpp-2.
// ...existing code...

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    // ...existing code...

    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;
    let mut consecutive_eos = 0;
    
    let t_main_start = ggml_time_us();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::dist(seed.unwrap_or(1234)),
        LlamaSampler::greedy(),
    ]);

    while n_cur <= n_len {
        // sample the next token 
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        // Check for EOS loop
        if model.is_eog_token(token) {
            consecutive_eos += 1;
            if consecutive_eos >= 3 {
                eprintln!("Breaking out of EOS loop");
                break;
            }
        } else {
            consecutive_eos = 0;
        }

        let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;
        let mut output_string = String::with_capacity(32);
        let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
        
        if !output_string.is_empty() {
            print!("{output_string}");
            std::io::stdout().flush()?;
        }

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;

        // Evaluate with error handling
        if let Err(e) = ctx.decode(&mut batch) {
            eprintln!("Decode error: {}", e);
            break;
        }

        n_decode += 1;
    }

    // ...existing code...
    Ok(())
}
