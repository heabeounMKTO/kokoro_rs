mod kokoro_core;
mod kokoro_utils;
use kokoro_utils::{normalize_text, pad_token_vec_with_zero, save_wav, save_wav_scalar};
use kokoro_core::{KokoroInput, KokoroModel, KokoroQuantLevel, EN_G2P, MAX_PHONEME_LENGTH, VOCAB};
use clap::Parser;
use std::time::Instant;
use tokenizers::tokenizer::{Tokenizer};
use ndarray::{concatenate, Array1, Axis};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Parser, Debug)]
struct CliArgs {
    #[arg(long)]
    model: String,

    #[arg(long)]
    input_text: String,

    #[arg(long)]
    voice_name: Option<String>
} 

const SPLASH: &str = r#"
    __         __                             ___ 
   / /______  / /______  _________      _____/ (_)
  / //_/ __ \/ //_/ __ \/ ___/ __ \    / ___/ / / 
 / ,< / /_/ / ,< / /_/ / /  / /_/ /   / /__/ / /  
/_/|_|\____/_/|_|\____/_/   \____/____\___/_/_/   
                                /_____/           "#;

fn split_into_chunks(input: &str, max_tokens: i64) -> Vec<String> {
    // Define sentence-ending punctuation and breaks
    let sentence_boundaries = ['.', '!', '?', '\n'];
    
    // First, split by obvious sentence boundaries
    let initial_splits: Vec<&str> = input
        .split(|c| sentence_boundaries.contains(&c))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_token_count: i64 = 0;
    
    // Estimate tokens (rough approximation: every 4 chars is about 1 token)
    let estimate_tokens = |text: &str| -> i64 {
        (text.chars().count() as i64 + 3) / 4
    };

    for sentence in initial_splits {
        // If the sentence alone might exceed token limit, split it further
        if estimate_tokens(sentence) > max_tokens {
            // Split into words while preserving meaningful boundaries
            let words: Vec<&str> = sentence
                .split_word_bounds()
                .collect();

            for word in words {
                let word_tokens = estimate_tokens(word);
                
                // If adding this word would exceed the limit, start a new chunk
                if current_token_count + word_tokens > max_tokens && !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                    current_chunk = String::new();
                    current_token_count = 0;
                }
                
                // Add word to current chunk
                current_chunk.push_str(word);
                current_token_count += word_tokens;

                // If we hit a natural pause point (comma, semicolon, etc.)
                if word.ends_with([',', ';', ':', '-']) && current_token_count > (max_tokens / 2) {
                    chunks.push(current_chunk.trim().to_string());
                    current_chunk = String::new();
                    current_token_count = 0;
                }
            }
        } else {
            // If this sentence would push us over the limit, start a new chunk
            if current_token_count + estimate_tokens(sentence) > max_tokens && !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk = String::new();
                current_token_count = 0;
            }
            
            // Add the sentence
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence);
            current_token_count += estimate_tokens(sentence);
        }
    }

    // Don't forget the last chunk
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    chunks
}

fn main() {
    println!("{}", SPLASH);
    let args = CliArgs::parse();
    let voice_name: String = match &args.voice_name {
        Some(voice) => String::from(voice),
        None => String::from("af_heart")
    };
    let kokoro_model = KokoroModel::load_model(
        &args.model,
        KokoroQuantLevel::Fp32,
    );
    let t1 = Instant::now();
    let chunk_len = 1;
    if &args.input_text.len() <= &chunk_len {
        let input2 = KokoroInput::new(&args.input_text, &voice_name, 1.0);
        println!("[DEBUG] KOKORO INPUT {:?}", input2);
        let reshape_result2 = &kokoro_model.generate(&input2).unwrap();
        println!("[DEBUG] model: {:?}\ninference time: {:?}", &args.model ,t1.elapsed());
        save_wav(&reshape_result2, 22500, "output.wav").unwrap();
    } else {

        let _all_splits = split_into_chunks(&args.input_text, 128);
        let mut all_w: Vec<Vec<f32>> = vec![];
        for (idx, _splits) in _all_splits.iter().enumerate() {
            let input2 = KokoroInput::new(&_splits, &voice_name, 1.0);
            println!("[DEBUG] KOKORO INPUT {:?}", input2);
            let reshape_result2 = kokoro_model.generate(&input2).unwrap().to_vec();
            all_w.push(reshape_result2);
            println!("[DEBUG] model: {:?}\ninference time: {:?}", &args.model ,t1.elapsed());
            // save_wav(&reshape_result2, 22500, &format!("output_chunk{}.wav", idx)).unwrap();
        } 
        let mut final_a : Vec<f32> = vec![];
        for sub_vec in &all_w {
            final_a.extend(sub_vec);
        }
        save_wav_scalar(final_a, 22500, "output_combined.wav").unwrap();
    }
}
