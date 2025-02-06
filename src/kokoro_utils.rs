use crate::kokoro_core::EN_G2P;
use hound;
use ndarray::{Array1, ArrayBase, Axis, Dim, OwnedRepr, ViewRepr};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::read;

/// writing HashMap<String, ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>>  every fucking time is
/// CRAZY
/// TODO: refactor to a [510, 1, 256] instead of some dynamic array bullshit
#[derive(Debug)]
pub struct KokoroVoice {
    pub styles: HashMap<String, ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>>,
}

pub fn save_wav(data: &Array1<f32>, sample_rate: u32, filename: &str) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(filename, spec)?;

    for &sample in data {
        // Convert float to 16-bit integer
        let scaled = (sample * 32767.0) as i16;
        writer.write_sample(scaled)?;
    }

    Ok(())
}

pub fn string_to_tokens(text: &str, vocab: &HashMap<char, usize>) -> Vec<usize> {
    let _nrm_text = normalize_text(&text.to_owned());
    let _g2p = &EN_G2P
        .convert_to_phonemes(
            &_nrm_text,
            lazy_phonememize::phonememizer::PhonemeOutputType::ASCII,
        )
        .unwrap();
    println!("[DEBUG] G2P {}", &_g2p);
    let tokens: Vec<usize> = _g2p
        .chars()
        .filter_map(|c| vocab.get(&c))
        .copied()
        .collect();
    println!("[DEBUG] STRING TO TOKENS {:?}", tokens);

    // add 0 in first and last element for padding
    let mut final_v = Vec::with_capacity(tokens.len() + 2);
    final_v.push(0);
    final_v.extend(tokens);
    final_v.push(0);
    final_v
}

pub fn kokoro_read_voice_vectors(voice_path: &str) -> KokoroVoice {
    let data = read(voice_path).unwrap();
    let tensors = SafeTensors::deserialize(&data).unwrap();
    let mut voice_mappings: HashMap<String, ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>> =
        HashMap::new();
    for (_name, tnsr) in tensors.tensors() {
        // let mut array = Vec::with_capacity(tnsr.shape());
        let data: Vec<f32> = bytemuck::cast_slice(tnsr.data()).to_vec();
        let  array = ArrayBase::from_shape_vec([510, 1, 256], data).unwrap();
        voice_mappings.insert(String::from(_name), array);
    }
    KokoroVoice {
        styles: voice_mappings,
    }
}

pub fn kokoro_select_voice(
    token_index: usize,
    voice: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>,
) -> ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>> {
    let select_voice = voice.index_axis(Axis(0), token_index); // `string_to_token` accounts for +2 padding, so 510 + 2 (0 front + 0 back)
    select_voice
}

pub fn normalize_text(text: &str) -> String {
    let mut result: String = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.trim().to_string())
        .collect::<Vec<String>>()
        .join("\n");
    result = result
        .replace('\u{2018}', "'") // Left single quote
        .replace('\u{2019}', "'") // Right single quote
        .replace("«", "\u{201C}") // Left double quote
        .replace("»", "\u{201D}") // Right double quote
        .replace('\u{201C}', "\"") // Left double quote
        .replace('\u{201D}', "\"") // Right double quote
        .replace("(", "«")
        .replace(")", "»");
    let replacements = [
        ("、", ", "),
        ("。", ". "),
        ("！", "! "),
        ("，", ", "),
        ("：", ": "),
        ("；", "; "),
        ("？", "? "),
    ];

    for (from, to) in replacements.iter() {
        result = result.replace(from, to);
    }
    while result.contains("  ") {
        result = result.replace("  ", " ");
    }

    while result.contains(" \n") || result.contains("\n ") {
        result = result.replace(" \n", "\n").replace("\n ", "\n");
    }

    let title_replacements = [
        (" Dr. ", " Doctor "),
        (" MR. ", " Mister "),
        (" Mr. ", " Mister "),
        (" MS. ", " Miss "),
        (" Ms. ", " Miss "),
        (" MRS. ", " Mrs "),
        (" Mrs. ", " Mrs "),
    ];

    for (from, to) in title_replacements.iter() {
        result = result.replace(from, to);
    }

    result = result.replace(" etc.", " etc");

    result = result.replace("yeah", "ye'a").replace("Yeah", "Ye'a");

    let mut parts: Vec<String> = result
        .split(' ')
        .map(|word| {
            if word.contains("$") || word.contains("£") {
                // Simple money handling
                word.replace("$", "dollars ").replace("£", "pounds ")
            } else if word.contains("-") && word.chars().any(|c| c.is_digit(10)) {
                // Replace hyphens between numbers with "to"
                word.replace("-", " to ")
            } else {
                word.to_string()
            }
        })
        .collect();

    for i in 0..parts.len() {
        if parts[i].ends_with("'s") && parts[i].len() > 2 {
            let last_char = parts[i].chars().nth(parts[i].len() - 3).unwrap();
            if last_char.is_uppercase() && !"AEIOU".contains(last_char) {
                parts[i] = format!("{}'{}", &parts[i][..parts[i].len() - 2], "S");
            }
        }
    }
    result = parts.join(" ");
    result.trim().to_string()
}
