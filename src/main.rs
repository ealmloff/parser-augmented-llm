use fundu::parse_duration;
use futures_util::{pin_mut, stream, Stream, StreamExt};
use kalosm::language::*;
use libflate::gzip::MultiDecoder;
use reqwest::Client;
use reqwest::{header::RANGE, Url};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedReceiver};
use tokio::sync::RwLock;
use tokio::{fs::File, io::AsyncWriteExt};
use types::{Chat, Train};
use urlencoding::encode;
use warc::WarcHeader;
use warc::WarcReader;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dataset = Dataset::new().await;

    #[derive(Debug, Clone, Schema, Parse, Serialize, Deserialize)]
    struct SimilarityScore {
        /// A summary of the first text
        first_text_summary: String,
        /// A summary of the second text
        second_text_summary: String,
        /// A similarity score between the two texts between 0 and 100 where 0 is completely different ideas and 100 is complete correlated ideas
        similarity: u8,
    }

    let mut tools = Tools::new();
    tools.add_function(Function::new(
        "respond_with_captions",
        "Report information about the structure of the two text inputs",
        SimilarityScore::schema(),
    ));
    let prompt = "You are a content analysis expert. You take two pieces of text and identify if they are talking about the same idea, or not. You will call the score_similarity function exactly once for each pair of texts.";

    let (input_sender, input_receiver) = tokio::sync::mpsc::unbounded_channel();

    let mut results = dataset.generate(prompt, input_receiver, tools);

    tracing_subscriber::fmt::init();

    let indicies = [
        // "CC-MAIN-2024-22",
        // "CC-MAIN-2024-18",
        // "CC-MAIN-2024-10",
        // "CC-MAIN-2023-50",
        // "CC-MAIN-2018-30",
        "CC-MAIN-2024-26",
    ];

    println!("Enter a URL to search for:");
    let mut input_lines = BufReader::new(tokio::io::stdin()).lines();

    let mut visited: HashMap<String, u64> = HashMap::new();
    let mut visited_urls: HashSet<String> = HashSet::new();

    while let Some(url) = input_lines.next_line().await? {
        for index in indicies {
            let cc = CommonCrawl::new(index.to_string());
            let records = cc.search(&url).await?;
            pin_mut!(records);

            while let Some(range) = records.next().await {
                let Some(mut url) = range.url() else {
                    continue;
                };
                url.set_fragment(None);
                url.set_query(None);
                if !visited_urls.insert(url.to_string()) {
                    continue;
                }
                let domain = url.domain().unwrap_or("unknown_domain").to_string();
                let entry = visited.entry(domain.clone()).or_default();
                let threshold = 500;
                if *entry > threshold {
                    println!("the {domain} domain has been visited more than {threshold} times, skipping");
                    continue;
                }
                *entry += 1;
                if let Some(slice) = cc.fetch_slice(&range).await? {
                    for page in slice.iter() {
                        let Ok(html) = page.json() else {
                            continue;
                        };
                        // put the html in the results folder
                        let mut path = url.path().to_string();
                        if path.ends_with('/') {
                            path += "index";
                        }
                        let path = format!("./results/{domain}{path}");
                        let path = PathBuf::from(path);
                        println!(
                            "Writing {} to {}",
                            page.url().unwrap().path(),
                            path.display()
                        );
                        if std::fs::create_dir_all(path.parent().unwrap()).is_ok() {
                            if let Ok(mut file) = std::fs::File::create(path) {
                                let _ = file.write_all(html.to_string().as_bytes());
                            }
                        }
                    }
                }
            }
        }
        println!("Enter another URL to search for:");
    }

    tokio::spawn(async move {
        while let Some(train) = results.recv().await {
            println!("{}", train.response.model);
            let message = &train.response.choices[0].message;
            println!("{}", message.content);
            for tool_call in &message.tool_calls {
                println!("{}", tool_call.function.name);
                if let Ok(character) =
                    serde_json::from_str::<SimilarityScore>(&tool_call.function.arguments)
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
        let first_text = prompt_input("first: ").unwrap();
        let second_text = prompt_input("second: ").unwrap();
        let input =
            format!("Summarize each of the following texts and identify if they are referring to the same topic:\nFIRST TEXT:\n\"{first_text}\"\nSECOND TEXT:\n\"{second_text}\"");

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

const DATA_URL: &str = "https://data.commoncrawl.org/";

const MAX_DURATION: std::time::Duration = std::time::Duration::from_secs(120);

// Retry every 1 second. Common crawl is being ddos'd. https://commoncrawl.org/blog/oct-nov-2023-performance-issues
async fn try_any_with_backoff<F: Future<Output = Result<T, reqwest::Error>>, T>(
    mut f: impl FnMut() -> F,
) -> Result<T, reqwest::Error> {
    let mut backoff = std::time::Duration::from_secs(1);
    loop {
        match f().await {
            Ok(response) => {
                return Ok(response);
            }
            Err(e) => {
                let status = e.status();
                match status {
                    Some(
                        reqwest::StatusCode::TOO_MANY_REQUESTS
                        | reqwest::StatusCode::SERVICE_UNAVAILABLE,
                    ) => {
                        if backoff > MAX_DURATION {
                            return Err(e);
                        }
                        println!("Retrying request...");
                    }
                    _ => return Err(e),
                }
            }
        }
        tokio::time::sleep(backoff).await;
        backoff *= 2;
    }
}

async fn try_with_backoff<F: Future<Output = Result<reqwest::Response, reqwest::Error>>>(
    mut f: impl FnMut() -> F,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut backoff = std::time::Duration::from_secs(1);
    loop {
        match f().await {
            Ok(response) => {
                let status = response.status();
                if status == reqwest::StatusCode::TOO_MANY_REQUESTS
                    || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
                {
                    println!("Retrying request...");
                } else {
                    return Ok(response);
                }
            }
            Err(e) => {
                let status = e.status();
                match status {
                    Some(
                        reqwest::StatusCode::TOO_MANY_REQUESTS
                        | reqwest::StatusCode::SERVICE_UNAVAILABLE,
                    ) => {
                        if backoff > MAX_DURATION {
                            return Err(e);
                        }
                        println!("Retrying request...");
                    }
                    _ => return Err(e),
                }
            }
        }
        tokio::time::sleep(backoff).await;
        backoff *= 2;
    }
}

#[derive(Deserialize, Debug)]
struct Record {
    filename: String,
    #[serde(deserialize_with = "deserialize_usize_from_string")]
    offset: usize,
    #[serde(deserialize_with = "deserialize_usize_from_string")]
    length: usize,
    languages: String,
    url: String,
}

impl Record {
    fn url(&self) -> Option<Url> {
        Url::parse(&self.url).ok()
    }
}

fn deserialize_usize_from_string<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

struct CommonCrawl {
    server: Url,
    index_name: String,
}

impl CommonCrawl {
    fn new(index_name: String) -> Self {
        const CC_INDEX_SERVER: &str = "http://index.commoncrawl.org/";
        Self {
            server: Url::parse(CC_INDEX_SERVER).unwrap(),
            index_name,
        }
    }

    fn latest() -> Self {
        Self::new("CC-MAIN-2024-26".to_string())
    }

    async fn search(&self, url: &str) -> Result<impl Stream<Item = Record>, reqwest::Error> {
        let encoded_url = encode(url);
        let pages_url = format!(
            "{}{}-index?url={}&showNumPages=true",
            self.server, self.index_name, encoded_url
        );

        #[derive(Deserialize)]
        struct PageInfo {
            pages: usize,
        }

        let page_info: PageInfo = try_any_with_backoff(|| async {
            let response = try_with_backoff(|| reqwest::get(&pages_url)).await?;
            response.json().await
        })
        .await?;

        let pages = futures_util::stream::iter(0..page_info.pages);

        let pages_url = format!(
            "{}{}-index?url={}",
            self.server, self.index_name, encoded_url
        );

        let stream = pages.flat_map(move |page| {
            println!("Fetching page {}/{}", page + 1, page_info.pages);
            let index_url = format!("{pages_url}&output=json&page={page}",);

            stream::once(try_any_with_backoff(move || {
                let index_url = index_url.clone();
                async move {
                    let response = try_with_backoff(|| reqwest::get(&index_url)).await?;
                    response.text().await
                }
            }))
            .flat_map(|response| {
                stream::iter(
                    response
                        .into_iter()
                        .flat_map(|response| {
                            response
                                .lines()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                        })
                        .flat_map(|line| serde_json::from_str::<Record>(&line).ok())
                        .filter(|record| record.languages.contains("eng")),
                )
            })
        });

        Ok(stream)
    }

    async fn fetch_slice(&self, record: &Record) -> Result<Option<WarcSlice>, reqwest::Error> {
        let s3_url = format!("{DATA_URL}{}", record.filename);

        let client = reqwest::Client::new();
        let fetch = || async {
            client
                .get(&s3_url)
                .header(
                    RANGE,
                    format!(
                        "bytes={}-{}",
                        record.offset,
                        record.offset + record.length - 1
                    ),
                )
                .send()
                .await
        };
        let response = try_with_backoff(fetch).await?;

        Ok(
            if response.status() == reqwest::StatusCode::PARTIAL_CONTENT {
                let bytes = response.bytes().await?.to_vec();
                Some(WarcSlice::new(bytes))
            } else {
                tracing::error!("Failed to fetch data: {:?}", response);
                None
            },
        )
    }
}

struct WarcSlice {
    contents: Vec<u8>,
}

impl WarcSlice {
    fn new(contents: Vec<u8>) -> Self {
        Self { contents }
    }

    fn iter(&self) -> impl Iterator<Item = Page> + '_ {
        let decoder = MultiDecoder::new(self.contents.as_slice()).unwrap();
        let warc = WarcReader::new(std::io::BufReader::with_capacity(1_048_576, decoder));
        warc.iter_records().flatten().map(Page::new)
    }
}

struct Page {
    record: warc::Record<warc::BufferedBody>,
}

impl Page {
    fn new(record: warc::Record<warc::BufferedBody>) -> Self {
        Self { record }
    }

    fn url(&self) -> Option<Url> {
        Url::parse(&self.record.header(WarcHeader::TargetURI)?).ok()
    }

    fn json(&self) -> anyhow::Result<serde_json::Value> {
        let body = self.record.body();
        let as_str = std::str::from_utf8(body)?;

        Ok(serde_json::from_str(as_str)?)
    }
}
