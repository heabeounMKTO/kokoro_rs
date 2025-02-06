mod kokoro_core;
mod kokoro_utils;
use kokoro_utils::save_wav;
use kokoro_core::{KokoroInput, KokoroModel, KokoroQuantLevel};
use clap::Parser;
use std::time::{self, Instant};

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
    let input2 = KokoroInput::new(&args.input_text, &voice_name, 1.0);
    let reshape_result2 = &kokoro_model.generate(&input2).unwrap();
    println!("[DEBUG] model: {:?}\ninference time: {:?}", &args.model ,t1.elapsed());
    save_wav(&reshape_result2, 22500, "output.wav").unwrap();
}
