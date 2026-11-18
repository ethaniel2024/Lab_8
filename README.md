# STAT 386 Lab 8: Intro to Machine Learning

This lab continues the cumulative STAT 386 repository. Use the cleaned General Conference talk dataset produced in the previous labs to conduct exploratory analysis and communicate one finding with a polished figure.

## Create your personal repository

1. Click **Use this template** and select **Create a new repository**.
2. For **Owner**, select `Stat386-Fall-2026`.
3. Name your repository using the format `your_netid_lab_8`.
4. Set the repository visibility to **Private**.
5. Click **Create repository**.

## Clone and set up the project

Clone your personal repository inside your `STAT_386` folder:

```bash
git clone <your-repository-url>
cd <your-repository-name>
uv sync
```

The first time you run `uv sync`, it will create a `uv.lock` file. Include that file in your commits.

Open the repository folder in Positron. Select the Python interpreter inside the `.venv` folder created by `uv`.


## Complete the lab

Open `lab_8.qmd` and replace `Your Name` with your name. Work through the sections in order.

Begin by loading the provided October 2025 General Conference talk dataset and shared embedding cache:

- `data/processed/talks_october_2025_cleaned.csv`
- `data/processed/talk_embeddings.npz`
- `data/processed/talk_embeddings_metadata.csv`

The shared embedding cache contains vectors for 33 regular conference talks, generated using the `nomic-embed-text:v1.5` model. You do not need to scrape the website or regenerate the talk embeddings. The cache has already been generated and validated for this lab.

Complete the data and embedding checks, examine the structure of the embedding space, and use PCA and UMAP to explore how the talks relate to one another. Then formulate your own gospel-related question, generate an embedding for that question using the same model, and use cosine similarity to identify the three most relevant talks.

**Data input:** Use the cleaned dataset and shared embedding files provided in `data/processed/`. Do not replace these files with raw scraped data or attempt to regenerate the shared talk embeddings.

**Query embedding:** You will need Ollama and the `nomic-embed-text:v1.5` model installed locally to generate an embedding for your own question. Use the required `search_query:` prefix when embedding your question. The shared talk embeddings use the corresponding `search_document:` prefix.

You can preview your document with:

    uv run quarto preview lab_8.qmd

Stop the preview by pressing `Ctrl+C` in the terminal. Make several meaningful commits as you work. For example:

    git add lab_8.qmd
    git commit -m "Complete embedding exploration"
    git push

## Work with the AI Collaborator

After completing your embedding exploration and semantic search, commit and push your work. You will then review two proposed changes to your lab code. Each interaction creates a Pull Request.

Both reviews ask you to evaluate a proposed change related to the embedding analysis. Inspect the proposed code, consider its effect on the analysis and interpretation, and decide whether the change is appropriate.

You are evaluated on your reasoning and ability to interpret semantic relationships responsibly, not simply on whether you accept or reject a proposed change.

### First AI Collaborator interaction

Create a new Issue using the **Lab 8 Complete** Issue template. If the template is unavailable, create a regular Issue titled `Lab 8 complete`; its description can be empty.

Post the following command as a **new comment**, not in the Issue description:

    @local-llm-user process config-dir: Lab_8/metadata_01.yml in instructor-repo: Stat386-Fall-2026/Instructor_Repo

Once the collaborator creates a Pull Request:

1. Read the Pull Request description.
2. Inspect the proposed code in **Files changed**.
3. Answer every review question in a comment on that Pull Request.
4. Explain whether the proposed addition fits the lab and supports a reasonable interpretation of the embedding results.
5. Merge the Pull Request if appropriate, or close the Pull Request without merging if inappropriate.

You are evaluated on your reasoning, not on accepting the change. You do not need to execute a proposed analysis simply to evaluate its design or interpretation.

If you merge the Pull Request, update your local repository before continuing:

    git pull

### Second AI Collaborator interaction

After reviewing the first Pull Request, return to the **same Lab 8 completion Issue**. If the Issue was closed automatically, reopen it before posting the next command.

Post the following command as a new comment:

    @local-llm-user process config-dir: Lab_8/metadata_02.yml in instructor-repo: Stat386-Fall-2026/Instructor_Repo

Review the second Pull Request using the same process. Answer all questions in a comment and explain your decision to merge or close it. If you merge it, run `git pull` again.

## Render and submit

After both reviews, render the final document from a fresh session:

    uv run quarto render lab_8.qmd

Commit and push your completed document, rendered HTML, dependency lockfile, and any other required work products:

    git add lab_8.qmd lab_8.html uv.lock
    git commit -m "Complete Lab 8"
    git push

Follow the submission directions provided in Canvas.

## Deliverables

- Completed `lab_8.qmd`.
- Completed data and embedding validation.
- PCA and UMAP exploration of the talk embeddings.
- One original gospel-related query embedded using the required model.
- The three most similar talks identified using cosine similarity.
- A projection of your query into the PCA and UMAP spaces.
- A concise interpretation and discussion of limitations.
- A response to both AI Collaborator reviews.

This is an exploratory analysis of semantic similarity. Similarity scores indicate how closely the model represents two pieces of text in its embedding space; they do not establish doctrinal correctness, prove that a talk answers your question, or measure the spiritual importance of a topic. Interpret the results thoughtfully and acknowledge the limitations of using an embedding model to represent religious discourse.
