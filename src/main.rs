use fundu::parse_duration;
use kalosm::language::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedReceiver};
use tokio::sync::RwLock;
use tokio::{fs::File, io::AsyncWriteExt};
use types::{Chat, Train};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dataset = Dataset::new().await;

    #[derive(Debug, Clone, Schema, Parse, Serialize, Deserialize)]
    struct Companies {
        /// A list of companies that may use software
        companies: Vec<Company>,
    }

    #[derive(Debug, Clone, Schema, Parse, Serialize, Deserialize)]
    struct Company {
        /// A fake name for the company. The name should be multiple words
        name: String,
        /// A brief description of the company
        description: String,
        /// An longer description of how the company may use software
        analysis: String,
    }

    let mut tools = Tools::new();
    tools.add_function(Function::new(
        "submit_companies",
        "Report a list of companies that may use software",
        Companies::schema(),
    ));
    let prompt = "You are a contractor tasked with generating a list of potential candidates that may use your company's software expertise. You will call the submit_companies function exactly once with the list of all companies that may use your software.";

    let (input_sender, input_receiver) = tokio::sync::mpsc::unbounded_channel();

    let mut results = dataset.generate(prompt, input_receiver, tools);

    tracing_subscriber::fmt::init();

    tokio::spawn(async move {
        while let Some(train) = results.recv().await {
            println!("{}", train.response.model);
            let message = &train.response.choices[0].message;
            println!("{}", message.content);
            for tool_call in &message.tool_calls {
                println!("{}", tool_call.function.name);
                if let Ok(character) =
                    serde_json::from_str::<Companies>(&tool_call.function.arguments)
                {
                    println!("\n\n{:#?}\n\n", character);
                } else {
                    println!("Error parsing tool call");
                    println!("{}", tool_call.function.arguments);
                }
            }
        }
    });

    loop {
        let input =
            "Generate a list of companies that may use your company's software expertise. Remember, many traditional companies use some software. For example, a good company would be grocery store or marketing company. Respond with at least 100 companies".to_string();

        input_sender.send(input).unwrap();
    }
}

#[derive(Debug, Copy, Clone)]
struct Provider {
    url: &'static str,
    key: &'static str,
    models: &'static [types::Model],
}

const PROVIDERS: &[Provider] = &[CEREBRAS_PROVIDER, GROQ_PROVIDER];

const CEREBRAS_PROVIDER: Provider = Provider {
    url: "https://api.cerebras.ai/v1/chat/completions",
    key: std::env!("CEREBRAS_KEY"),
    models: &[types::Model::Llama3_70b, types::Model::Llama3_8b],
};

const GROQ_PROVIDER: Provider = Provider {
    url: "https://api.groq.com/openai/v1/chat/completions",
    key: std::env!("GROQ_KEY"),
    models: &[types::Model::Llama3Groq8b, types::Model::Llama3Groq70b],
};

struct Dataset {
    client: Arc<Client>,
    responses: Arc<RwLock<Vec<Train>>>,
    data_count: Arc<AtomicUsize>,
    models_sender: Sender<(types::Model, Provider)>,
    models: Receiver<(types::Model, Provider)>,
}

impl Dataset {
    async fn new() -> Self {
        let client = Client::new();
        let (models_sender, models) = tokio::sync::mpsc::channel(100);
        for provider in PROVIDERS {
            for &model in provider.models {
                models_sender.send((model, *provider)).await.unwrap();
            }
        }
        Self {
            client: Arc::new(client),
            responses: Arc::new(RwLock::new(Vec::new())),
            data_count: Arc::new(AtomicUsize::new(0)),
            models_sender,
            models,
        }
    }

    fn generate(
        mut self,
        prompt: &str,
        mut input: UnboundedReceiver<String>,
        tools: Tools,
    ) -> UnboundedReceiver<Train> {
        let (train_sender, train_receiver) = tokio::sync::mpsc::unbounded_channel();
        let prompt = prompt.to_string();

        tokio::spawn(async move {
            loop {
                let (model, provider) = self.models.recv().await.unwrap();

                let data_count = self.data_count.clone();
                let responses = self.responses.clone();
                let sender = self.models_sender.clone();
                let client = self.client.clone();
                let prompt = prompt.clone();
                let input = input.recv().await.unwrap();
                let tools = tools.clone();
                let train_sender = train_sender.clone();

                tokio::spawn(async move {
                    loop {
                        match try_generate(&client, model, &tools, &prompt, &input, provider).await
                        {
                            Ok(Some(train)) => {
                                sender.send((model, provider)).await.unwrap();
                                train_sender.send(train.clone()).unwrap();
                                Dataset::add_response(responses, &data_count, train).await;
                                break;
                            }
                            Ok(None) => {}
                            Err(e) => println!("Error: {}", e),
                        }
                    }
                });
            }
        });

        train_receiver
    }

    async fn add_response(
        responses: Arc<RwLock<Vec<Train>>>,
        data_count: &AtomicUsize,
        train: Train,
    ) {
        {
            RwLock::write(&responses).await.push(train);
        }

        // It was successful, so we can move on to the next component
        let data_count = data_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if data_count % 10 == 0 {
            // write the data into a file
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let folder = PathBuf::from("../data");
            if !folder.exists() {
                std::fs::create_dir_all(&folder).unwrap();
            }
            let file = folder.join(format!("data{}.json", timestamp));
            let mut file = File::create(file).await.unwrap();
            let mut responses = RwLock::write(&responses).await;
            let json = serde_json::to_string(&*responses).unwrap();
            file.write_all(json.as_bytes()).await.unwrap();
            responses.clear();
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Function {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl Function {
    fn new(name: &str, description: &str, parameters: SchemaType) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            parameters: serde_json::from_str(&parameters.to_string()).unwrap(),
        }
    }
}

#[derive(Clone)]
struct Tools {
    functions: Vec<Function>,
}

impl Tools {
    fn new() -> Self {
        Self {
            functions: Vec::new(),
        }
    }

    fn add_function(&mut self, function: Function) {
        self.functions.push(function);
    }

    fn to_json(&self) -> serde_json::Value {
        self.functions
            .iter()
            .map(|function| {
                serde_json::json!(
                    {
                        "type": "function",
                        "function": function
                    }
                )
            })
            .collect()
    }
}

fn smart_model_chat_history(prompt: String, input: String) -> serde_json::Value {
    serde_json::json!([
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": input
        }
    ])
}

fn model_chat_history(_model: types::Model, prompt: String, input: String) -> serde_json::Value {
    smart_model_chat_history(prompt, input)
}

async fn try_generate(
    client: &Client,
    model: types::Model,
    tools: &Tools,
    prompt: &str,
    input: &str,
    provider: Provider,
) -> Result<Option<Train>, reqwest::Error> {
    println!("Generating response with model: {}", model);
    let json = serde_json::json!({
        "messages": model_chat_history(model, prompt.to_string(), input.to_string()),
        "tools": tools.to_json(),
        "tool_choice": "required",
        "model": model,
        "seed": rand::random::<u32>(),
        "model": model,
        "temperature": 0.5,
        "top_p": 0.65,
    });
    println!("{}", json);
    let res = match client
        .post(provider.url)
        .header("Authorization", format!("Bearer {}", provider.key))
        .header("Content-Type", "application/json")
        .json(&json)
        .send()
        .await
    {
        Ok(res) => res,
        Err(e) => {
            println!("Request failed, sleeping until reset");
            println!("{}", e);
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            return Ok(None);
        }
    };

    let remaining_requests: u64 = res
        .headers()
        .get("x-ratelimit-remaining-requests")
        .and_then(|x| x.to_str().ok())
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    let remaining_tokens: u64 = res
        .headers()
        .get("x-ratelimit-remaining-tokens")
        .and_then(|x| x.to_str().ok())
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    let reset_requests = res
        .headers()
        .get("x-ratelimit-reset-requests")
        .and_then(|x| x.to_str().ok());
    let reset_requests = reset_requests
        .and_then(|reset_requests| parse_duration(reset_requests).ok())
        .unwrap_or(std::time::Duration::from_secs(15));
    let reset_tokens = res
        .headers()
        .get("x-ratelimit-reset-tokens")
        .and_then(|x| x.to_str().ok());
    let reset_tokens = reset_tokens
        .and_then(|reset_tokens| parse_duration(reset_tokens).ok())
        .unwrap_or(std::time::Duration::from_secs(15));
    println!("Remaining requests: {}", remaining_requests);
    println!("Remaining tokens: {}", remaining_tokens);
    println!("Reset requests: {:?}", reset_requests);
    println!("Reset tokens: {:?}", reset_tokens);

    if remaining_requests == 0 {
        println!("Rate limit reached, sleeping until reset");
        tokio::time::sleep(reset_requests).await;
        return Ok(None);
    }
    if remaining_tokens == 0 {
        println!("Rate limit reached, sleeping until reset");
        tokio::time::sleep(reset_tokens).await;
        return Ok(None);
    }

    if res.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        println!("Rate limit reached, sleeping until reset");
        tokio::time::sleep(reset_requests.max(reset_tokens)).await;
        return Ok(None);
    }

    let text = res.text().await?;

    let Ok(chat) = serde_json::from_str::<Chat>(&text) else {
        println!("Deserialization failed, sleeping until reset");
        println!("{}", text);
        tokio::time::sleep(reset_requests.max(reset_tokens)).await;
        return Ok(None);
    };

    let train = Train {
        prompt: prompt.to_string(),
        response: chat,
    };

    Ok(Some(train))
}
