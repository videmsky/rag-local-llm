import { RetrievalQAChain, loadQAStuffChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { Ollama } from "@langchain/community/llms/ollama";
import { PromptTemplate } from "@langchain/core/prompts";
// import { hey } from './helper.js';
// import yo from './helper.cjs';

// Ollama host URL
// const hostLocal = "http://localhost:11434";
const hostRemote = "https://o72mov3aflryk6-11434.proxy.runpod.net";

const loader = new CheerioWebBaseLoader(
  "https://nvie.com/posts/a-successful-git-branching-model"
);
const rawDocs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const docs = await splitter.splitDocuments(rawDocs);

const vectorstore = await MemoryVectorStore.fromDocuments(
  docs,
  new OllamaEmbeddings({model: "llama2", baseUrl: hostRemote})
);
const retriever = vectorstore.asRetriever();

const llm = new Ollama({
  model: "codellama",
  baseUrl: hostRemote
});

const template = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:`;

const QA_CHAIN_PROMPT = new PromptTemplate({
  inputVariables: ["context", "question"],
  template,
});

const chain = new RetrievalQAChain({
  combineDocumentsChain: loadQAStuffChain(llm, { prompt: QA_CHAIN_PROMPT }),
  retriever,
  returnSourceDocuments: true,
  inputKey: "question",
});

const response = await chain.call({
  question: "provide step by step instructions for creating a release branch using gitflow",
});

console.log(response);

// ===========
// console.log(hey);
// console.log(yo);