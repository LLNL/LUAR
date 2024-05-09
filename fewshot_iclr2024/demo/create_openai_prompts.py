
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

def create_openai_prompt(df_row, model="gpt-3.5-turbo"):
    title = df_row["title"]
    abstract = df_row["abstract"]
    # Prompt used by M4 for PeerRead: https://aclanthology.org/2024.eacl-long.83.pdf (Appendix A.3 as of 04/30/2024)
    prompt = "Write a peer review by first describing what problem or question this paper addresses, then strengths and weaknesses, for the paper {}, its main content is as below: {}.".format(title, abstract) 

    messages = [{"role":"system", "content":"You are a reviewer for the ICLR 2024 conference. You have been assigned to review the following paper: {}.".format(title)}, {"role":"user", "content":prompt}]

    return {
        "model": model,
        "messages": messages
    }

data = pd.read_json("iclr2024_20240430.json", lines=True)
openai_prompts = data.progress_apply(create_openai_prompt, axis=1)
with open("./openai_prompts.jsonl", "w") as f:
    f.write(openai_prompts.to_json(orient="records", lines=True))