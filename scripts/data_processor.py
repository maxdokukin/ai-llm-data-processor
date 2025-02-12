import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv('../.env')
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in the .env file.")
client = OpenAI(api_key=api_key)


def process_csv_with_openai(
    csv_filepath,
    prompt_filepath,
    output_csv_filepath,
    column_to_process,
    processed_column,
    model="gpt-4o-mini"
    ):

    with open(prompt_filepath, "r", encoding="utf-8") as f:
        prompt_template = f.read().strip()

    df = pd.read_csv(csv_filepath)

    responses = []
    for _, row in df.iterrows():
        cell_text = str(row[column_to_process])
        print("Processing:", cell_text)
        full_prompt = f"{prompt_template}\n{cell_text}"

        response = client.chat.completions.create(
            model=model,
            messages=[
                    {"role": "assistant", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ]
            )

        answer = response.choices[0].message.content
        responses.append(answer)
        print("Processed:", answer)

    df[processed_column] = responses

    df.to_csv(output_csv_filepath, index=False)
    print(f"Processed data saved to {output_csv_filepath}")


if __name__ == "__main__":
    process_csv_with_openai(
        csv_filepath="../data/raw/projects_fd.csv",
        prompt_filepath="../prompts/extract_skills_frameworks.txt",
        output_csv_filepath="../data/processed/projects_processed.csv",
        column_to_process="full_description_processed",           # Column to process in the CSV file
        processed_column="skills_frameworks"   # Name of the new column for API responses
    )
