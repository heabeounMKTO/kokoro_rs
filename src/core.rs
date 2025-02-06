use crate::kokoro_utils::{
    kokoro_read_voice_vectors, kokoro_select_voice, string_to_tokens, KokoroVoice,
};
use anyhow::Result;
use lazy_phonememize::phonememizer::convert_to_phonemes;
use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr, Array2, Array1};
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, TensorRTExecutionProvider},
    session::{builder::GraphOptimizationLevel, input, Session, SessionOutputs}, value::Tensor,
};
use std::collections::HashMap;

use ort::value::Value;

use lazy_static::lazy_static;
lazy_static! {
    pub static ref MAX_PHONEME_LENGTH: usize = 512;
    pub static ref VOCAB: HashMap<char, usize> = {
        let pad = "$";
        let punctuation = ";:,.!?¡¿—…\"«»\"\" ";
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        let letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

        let symbols: Vec<char> = format!("{}{}{}{}", pad, punctuation, letters, letters_ipa)
            .chars()
            .collect();

        let mut dicts = HashMap::new();
        for (i, symbol) in symbols.iter().enumerate() {
            dicts.insert(*symbol, i);
        }
        dicts
    };
    pub static ref VOICES: KokoroVoice = kokoro_read_voice_vectors("./models/voices.safetensors");
}

#[derive(Debug)]
pub struct KokoroModel {
    pub model: Session,
    pub is_fp16: bool,
}

impl KokoroModel {
    pub fn load_model(model_path: &str, fp_16: bool) -> KokoroModel {
        let _ = fp_16;
        let model: Session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_execution_providers([CUDAExecutionProvider::default().build(),CPUExecutionProvider::default().build()])
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();
        KokoroModel {
            model,
            is_fp16: false,
        }
    }

    pub fn warmup(&self) -> () {
        todo!()
    }

    pub fn generate(&self, input_data: KokoroInput) -> Result<Array1<f32>>{
        let _voice_tensor: Tensor<f32> = Tensor::from_array(input_data.voice).unwrap(); 
        let _token_tensor: Tensor<i64> = Tensor::from_array(([1, input_data.tokens.len()], input_data.tokens)).unwrap();
        let _speed_tensor: Tensor<f32> = Tensor::from_array(([1], vec![input_data.speed])).unwrap();
        todo!()
    }
}

#[derive(Debug)]
pub struct KokoroInput {
    pub tokens: Vec<i64>,
    pub voice: Array2<f32>, // do NOT panic its actually ArrayBase<OwnedRepr<f32> , Dim<[usize; 2]>> :) 
    pub speed: f32,
}

impl KokoroInput {
    pub fn new(input_string: &str, voice_name: &str, speed: f32) -> KokoroInput {
        let _tokens = string_to_tokens(input_string, &VOCAB);
        let selected_voice = kokoro_select_voice(_tokens.len(), &VOICES.styles.get(voice_name).unwrap())
                .to_owned().into_shape_with_order((1, 256)).unwrap();
        KokoroInput {
            tokens: _tokens.iter().map(|&x| x as i64).collect(),
            voice: selected_voice,
            speed: speed,
        }
    }
}
