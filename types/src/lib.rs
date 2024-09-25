use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Eq, Hash, PartialEq, Clone)]
pub struct Train {
    pub prompt: String,
    pub response: Chat,
}

#[derive(Deserialize, Serialize, Debug, Eq, Hash, PartialEq, Clone)]
pub struct Chat {
    #[serde(default)]
    pub model: Model,
    pub choices: Vec<Choice>,
}

#[derive(Deserialize, Serialize, Debug, Eq, Hash, PartialEq, Clone)]
pub struct Choice {
    pub message: Message,
}

#[derive(Deserialize, Serialize, Debug, Eq, Hash, PartialEq, Clone)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Deserialize, Serialize, Debug, Eq, Hash, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: Function,
}

#[derive(Deserialize, Serialize, Debug, Eq, Hash, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Logprobs {
    #[serde(default)]
    pub tokens: Vec<f64>,
    #[serde(default)]
    pub token_logprobs: Vec<f64>,
    #[serde(default)]
    pub top_logprobs: Vec<f64>,
    #[serde(default)]
    pub text_offset: Vec<usize>,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: usize,
    #[serde(default)]
    pub completion_tokens: usize,
    #[serde(default)]
    pub total_tokens: usize,
    #[serde(default)]
    pub prompt_time: f64,
    #[serde(default)]
    pub completion_time: f64,
    #[serde(default)]
    pub total_time: f64,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct XGroq {
    #[serde(default)]
    pub id: String,
}

#[derive(
    Deserialize, Serialize, Debug, Eq, Hash, PartialEq, Default, Copy, Clone, PartialOrd, Ord,
)]
pub enum Model {
    #[serde(rename = "gemma-7b-it")]
    Gemma1 = 0,
    #[serde(rename = "gemma2-9b-it")]
    Gemma = 1,
    #[serde(rename = "mixtral-8x7b-32768")]
    Mixtral = 2,
    #[serde(rename = "llama3-8b-8192")]
    Llama3_8b = 3,
    #[default]
    #[serde(rename = "llama3-70b-8192")]
    Llama3_70b = 4,
    #[serde(rename = "llama3-groq-8b-8192-tool-use-preview")]
    Llama3Groq8b = 5,
    #[serde(rename = "llama3-groq-70b-8192-tool-use-preview")]
    Llama3Groq70b = 6,
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama3_8b => write!(f, "llama3-8b-8192"),
            Self::Gemma => write!(f, "gemma2-9b-it"),
            Self::Gemma1 => write!(f, "gemma-7b-it"),
            Self::Mixtral => write!(f, "mixtral-8x7b-32768"),
            Self::Llama3_70b => write!(f, "llama3-70b-8192"),
            Self::Llama3Groq8b => write!(f, "llama3-groq-8b-8192-tool-use-preview"),
            Self::Llama3Groq70b => write!(f, "llama3-groq-70b-8192-tool-use-preview"),
        }
    }
}

impl Model {
    pub const ALL: &'static [Self] = &[
        Self::Gemma1,
        Self::Gemma,
        Self::Llama3_8b,
        Self::Mixtral,
        Self::Llama3_70b,
        Self::Llama3Groq8b,
        Self::Llama3Groq70b,
    ];

    pub const LLAMA: &'static [Self] = &[Self::Llama3_8b, Self::Llama3_70b];
}
