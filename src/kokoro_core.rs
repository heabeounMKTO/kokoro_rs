use crate::kokoro_utils::{
    kokoro_read_voice_vectors, kokoro_select_voice, string_to_phoneme, string_to_tokens, KokoroVoice
};
use anyhow::Result;
use lazy_phonememize::phonememizer::LazyPhonemizer;
use lazy_static::lazy_static;
use ndarray::{Array1, Array2, ArrayBase, Dim, IxDynImpl, OwnedRepr};
use ort::inputs;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, TensorRTExecutionProvider},
    session::{builder::GraphOptimizationLevel, input, Session, SessionOutputs},
    value::Tensor,
};
use std::collections::HashMap;
lazy_static! {
    pub static ref MAX_PHONEME_LENGTH: usize = 256;
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
    pub static ref EN_G2P: LazyPhonemizer = LazyPhonemizer::init(Some("en")).unwrap();
}

#[derive(Debug)]
pub enum KokoroQuantLevel {
    Fp32,
    Fp16,
    Int4,
}

#[derive(Debug)]
pub struct KokoroModel {
    pub model: Session,
    pub quant_level: KokoroQuantLevel,
}

impl KokoroModel {
    pub fn load_model(model_path: &str, quant_level: KokoroQuantLevel) -> KokoroModel {
        let model: Session = Session::builder()
            .unwrap()
            // .with_parallel_execution(true)
            // .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();
        let _model_load = KokoroModel { model, quant_level };
        println!("running warmup for model..");
        _model_load
            .generate(&KokoroInput::new("_dummy_input", "af_heart", 1.0))
            .unwrap();
        println!("warmup done!");
        _model_load
    }

    pub fn generate(&self, input_data: &KokoroInput) -> Result<Array1<f32>> {
        // currently there are no differences?!
        match self.quant_level {
            KokoroQuantLevel::Fp32 => {
                let _voice_tensor: Tensor<f32> =
                    Tensor::from_array(input_data.voice.to_owned()).unwrap();
                let _token_tensor: Tensor<i64> = Tensor::from_array((
                    [1, input_data.tokens.len()],
                    input_data.tokens.to_owned(),
                ))
                .unwrap();
                let _speed_tensor: Tensor<f32> =
                    Tensor::from_array(([1], vec![input_data.speed.to_owned()])).unwrap();
                let results = self
                .model
                .run(
                    inputs!["tokens" => _token_tensor, "style" => _voice_tensor, "speed" => _speed_tensor]
                        .unwrap(),
                )
                .unwrap();
                let extract_res = results["audio"].try_extract_tensor::<f32>().unwrap();
                Ok(extract_res
                    .to_owned()
                    .into_shape_with_order(extract_res.to_owned().shape()[0])
                    .unwrap())
            }
            _ => {
                todo!()
            }
        }
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
        let _tokens = string_to_tokens(input_string, &VOCAB, true);
        let selected_voice = kokoro_select_voice(
            _tokens.len().to_owned(),
            &VOICES.styles.get(voice_name).unwrap().to_owned(),
        )
        .to_owned()
        .into_shape_with_order((1, 256))
        .unwrap()
        .to_owned();
        KokoroInput {
            tokens: _tokens.iter().map(|&x| x as i64).collect(),
            voice: selected_voice,
            speed: speed,
        }
    }

    pub fn from_tokens(input_tokens: Vec<usize>, voice_name: &str, speed: f32) -> KokoroInput {
        let selected_voice = kokoro_select_voice(
            input_tokens.len().to_owned(),
            &VOICES.styles.get(voice_name).unwrap().to_owned(),
        )
        .to_owned()
        .into_shape_with_order((1, 256))
        .unwrap()
        .to_owned();
        KokoroInput {
            tokens: input_tokens.iter().map(|&x| x as i64).collect(),
            voice: selected_voice,
            speed: speed,
        }
    }

}
