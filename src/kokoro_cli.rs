mod kokoro_core;
mod kokoro_utils;
use kokoro_utils::{normalize_text, pad_token_vec_with_zero, save_wav, save_wav_scalar};
use kokoro_core::{KokoroInput, KokoroModel, KokoroQuantLevel, EN_G2P, MAX_PHONEME_LENGTH, VOCAB};
use clap::Parser;
use std::time::Instant;
use tokenizers::tokenizer::{Tokenizer};
use ndarray::{concatenate, Array1, Axis};


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
 __  __     ______     __  __     ______     ______     ______     ______     __         __    
/\ \/ /    /\  __ \   /\ \/ /    /\  __ \   /\  == \   /\  __ \   /\  ___\   /\ \       /\ \   
\ \  _"-.  \ \ \/\ \  \ \  _"-.  \ \ \/\ \  \ \  __<   \ \ \/\ \  \ \ \____  \ \ \____  \ \ \  
 \ \_\ \_\  \ \_____\  \ \_\ \_\  \ \_____\  \ \_\ \_\  \ \_____\  \ \_____\  \ \_____\  \ \_\ 
  \/_/\/_/   \/_____/   \/_/\/_/   \/_____/   \/_/ /_/   \/_____/   \/_____/   \/_____/   \/_/ 
"#;



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
    let chunk_len = 128;

    if &args.input_text.len() <= &chunk_len {
        let input2 = KokoroInput::new(&args.input_text, &voice_name, 1.0);
        let reshape_result2 = &kokoro_model.generate(&input2).unwrap();
        println!("[DEBUG] model: {:?}\ninference time: {:?}", &args.model ,t1.elapsed());
        save_wav(&reshape_result2, 22500, "output.wav").unwrap();
    } else {

        let pho = EN_G2P.convert_to_phonemes(&normalize_text(&args.input_text), 
                                lazy_phonememize::phonememizer::PhonemeOutputType::ASCII).unwrap();
        
        let _a : Vec<usize> = pho.chars().filter_map(|c| VOCAB.get(&c)).copied().collect(); 
        let _split_tokens = _a.chunks(*MAX_PHONEME_LENGTH).collect::<Vec<_>>(); 
        let mut final_wavs : Vec<Array1<f32>>= vec![];
        for _tokens in _split_tokens {
            // println!("[DEBUG] TOKENS {:?}", _tokens);
            let _input = KokoroInput::from_tokens(_tokens.to_owned(), &voice_name, 1.0);
            final_wavs.push(kokoro_model.generate(&_input).unwrap());
        }
        for (idx, _vec) in final_wavs.iter().enumerate() {
            save_wav(_vec, 22500, &format!("output_chunk{}.wav", idx)).unwrap();
        }
    }
}
