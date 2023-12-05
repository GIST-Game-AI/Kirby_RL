import os
import uuid
import boto3
from dotenv import load_dotenv
import openai
import json


load_dotenv()

# OpenAI API 키 설정
openai.api_key = "YOUR_API_KEY"
f = open("./prompt.txt", 'r')
whole = f.read()

client = openai.OpenAI()


def upload_file(file_path: str, folder_name: str = "Game_AI"):
    try:
        # s3 클라이언트 생성
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2",
            aws_access_key_id=os.environ["S3_ACCESS"],
            aws_secret_access_key=os.environ["S3_SECRET"],
        )
    except Exception as e:
        print(e)
    else:
        _, file_extension = os.path.splitext(file_path)
        random_file_name = f"{folder_name}/{str(uuid.uuid4()) + file_extension}"

        # Open the file and upload it
        with open(file_path, 'rb') as file:
            s3.upload_fileobj(file, "dangmuzi-photo", random_file_name)

        return "https://dangmuzi-photo.s3.ap-northeast-2.amazonaws.com/" + random_file_name


def get_response_from_gpt(img_path: str, game_state_prompt: str):
    # game_state ex -> Game: Kirby's Dream Land-(1st Platform Stage), Current State: { Kirby's Health: 5, Score: 200, Key-down: [Move: None, Action: None]}
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": "Suppose you are a Reinforcement Learning Agent in Kibrby's Dream Land game released by Nintendo in 1993"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": whole
                     },
                ]
            },
            {
                "role": "assistant",
                "content": "Okay, I will response as your instructed response example"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": game_state_prompt
                     },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": upload_file(img_path),
                        }
                    }
                ]
            }
        ],
        max_tokens=4000
    )

    result = response.choices[0].message.content
    return json.loads(result.strip("`""json""\n"))