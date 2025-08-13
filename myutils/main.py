from notion_client import Client
import datetime
from pathlib import Path
from typing import List
from myval import format_metrics, quantize_to_2dp
from mylog import Logger
import yaml
import argparse
import os
import tarfile
import zipfile
import uuid
import requests
import traceback

def load_notion_config(file_path='notion_key.yaml'):
    """load notion config from .yaml file
    Returns:
        tuple: (notion_token, database_id, platform_name)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            token = config.get('notion_token', '')
            db_id = config.get('database_id', '')
            pc_name = config.get('platform_name', '')
            return token, db_id, pc_name
    except FileNotFoundError:
        Logger.warning(f"⚠️ Warning, cannot find {file_path}")
        return '', '', ''
    except Exception as e:
        Logger.warning(f"⚠️ Warning, reading files with {e}")
        return '', '', ''

NOTION_TOKEN, DATABASE_ID, PLATFORM_NAME = load_notion_config()
PART_SIZE = 10 * 1024 * 1024

class FileUploader:
    def __init__(self, file, token=None):
        self.file = file
        self.token = token if token else NOTION_TOKEN
        self.content_type = "application/pdf"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }

        self.meta = None
        self.upload_id = None
        self.upload_url = None
        self.filename = None
        self.file_compress_path = None
        self.parts = None
        self.num_parts = None
        self.finished = None

    def upload(self):
        # step 1: compress
        tar_path = FileProcessor.compress_folder_to_tar(self.file)
        self.file_compress_path = tar_path
        self.filename = os.path.basename(tar_path)
        Logger.info(f"File compressed: {tar_path}")

        # step 2: chunkify
        self.parts = FileProcessor.chunkify(tar_path)
        self.num_parts = len(self.parts)
        Logger.info(f"File chunked into {self.num_parts} parts")

        # step 3: start multipart upload
        self.meta = self._create_multipart_upload(self.filename, self.num_parts)
        self.upload_id = self.meta["id"]
        self.upload_url = self.meta["upload_url"]  # includes /send endpoint
        Logger.info(f"Multipart upload initiated with ID: {self.upload_id}")

        # send each part
        for idx, part_path in self.parts:
            self._upload_part(self.upload_url, idx, part_path)
            Logger.info(f"Part {idx}/{self.num_parts} uploaded successfully")

        # complete
        self.finished = self._complete_multipart_upload(self.upload_id)
        Logger.info(f"File upload completed successfully")

    def attach_to_page(self, page_id, prefix_name=None):
        # attach to page
        filename = self.filename
        if prefix_name:
            filename = f"{prefix_name}-{self.filename}"
        result = self._append_file_to_page(page_id=page_id, filename=filename)
        Logger.info(f"File attached to page successfully: {filename}")
        self._clean_up()
        return result

    def _clean_up(self):
        # cleanup temp parts
        Logger.info(f"Cleaning up {len(self.parts)} + 1 temporary files")
        for _, p in self.parts:
            os.remove(p)
        os.remove(self.file_compress_path)
        Logger.info("Temporary files cleanup completed")

    def _create_multipart_upload(self, filename, num_parts):
        url = "https://api.notion.com/v1/file_uploads"
        payload = {
            "mode": "multi_part",
            "number_of_parts": num_parts,
            "filename": filename,
            "content_type": self.content_type,
        }
        resp = requests.post(url, headers=self.headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    def _upload_part(self, upload_url_base, part_num, part_path):
        url = f"{upload_url_base}"  # send endpoint includes id
        with open(part_path, 'rb') as pf:
            files = {
                'file': (os.path.basename(part_path), pf)
            }
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Notion-Version": "2022-06-28"
            }
            data = {
                'part_number': str(part_num)
            }
            resp = requests.post(url, headers=headers, files=files, data=data)
        resp.raise_for_status()

    def _complete_multipart_upload(self, upload_id):
        url = f"https://api.notion.com/v1/file_uploads/{upload_id}/complete"
        resp = requests.post(url, headers=self.headers, json={})
        resp.raise_for_status()
        return resp.json()

    def _append_file_to_page(self, page_id, filename):
        url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        payload = {
            "children": [
                {
                    "type": "file",
                    "file": {
                        "type": "file_upload",
                        "file_upload": {"id": self.upload_id},
                        "name": filename
                    }
                }
            ]
        }
        resp = requests.patch(url, headers=self.headers, json=payload)
        resp.raise_for_status()
        return resp.json()

class FileProcessor:
    @staticmethod
    def compress_folder_to_tar(folder_path):
        os.makedirs(".tmp", exist_ok=True)
        output_path = f".tmp/{uuid.uuid4()}"
        with tarfile.open(output_path, "w") as tar:
            tar.add(folder_path, arcname=os.path.basename(folder_path))
        return output_path

    @staticmethod
    def compress_folder_to_zip(folder_path):
        os.makedirs(".tmp", exist_ok=True)
        output_path = f".tmp/{uuid.uuid4()}"
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    abs_path = os.path.join(root, file)
                    # 保持压缩包内的相对路径
                    rel_path = os.path.relpath(abs_path, os.path.dirname(folder_path))
                    zipf.write(abs_path, arcname=rel_path)
        return output_path

    @staticmethod
    def chunkify(file_path, part_size=PART_SIZE):
        parts = []
        with open(file_path, 'rb') as f:
            index = 1
            while True:
                chunk = f.read(part_size)
                if not chunk:
                    break
                part_path = f"{file_path}.part{index}"
                with open(part_path, 'wb') as pf:
                    pf.write(chunk)
                parts.append((index, part_path))
                index += 1
        return parts

def build_table_block(metrics_table: List[str]) -> dict:
    table_data = list()
    for id, line in enumerate(metrics_table.split("\n")):
        if id == 1:
            continue
        table_data.append(line.split("|")[1:-1])
    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": len(table_data[0]),
            "has_column_header": True,
            "has_row_header": False,
            "children": [
                {
                    "object": "block",
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"type": "text", "text": {"content": cell}}] for cell in row
                        ]
                    }
                } for row in table_data
            ]
        }
    }

class ExpUpLoader:
    def __init__(self, exp_path, data_path, extra_tags, token=None, db_id=None, platform_name=None, **kwargs):
        self.exp_path = Path(exp_path)
        self.data_path = Path(data_path)

        exp_parts = self.exp_path.parts
        detect_index = exp_parts.index('detect')
        self.exp_name = "-".join(exp_parts[detect_index + 1:])
        self.data_name = self.data_path.stem

        self.notion_token = token if token else NOTION_TOKEN
        self.database_id = db_id if db_id else DATABASE_ID
        self.platform_name = platform_name if platform_name else PLATFORM_NAME

        self.tags = {self.data_name, self.platform_name, *extra_tags}

        Logger.info(f"Upload experiment: {self.exp_name}")
        Logger.info(f"Upload dataset: {self.data_name}")
        Logger.info(f"Upload tags: {self.tags}")

        self.metrics_dict = format_metrics(
            model_path=self.exp_path / "weights/best.pt",
            data_path=self.data_path,
            name='val/test',
            batch=8,
            imgsz=640
        )

        if not self.notion_token:
            raise ValueError("No Notion Token found, please set it in yaml file or pass it as parameter")
        if not self.database_id:
            raise ValueError("No Database ID found, please set it in yaml file or pass it as parameter")

        self.notion = Client(auth=self.notion_token)
        self.file_uploader = FileUploader(file=self.exp_path, token=self.notion_token)

    def upload(self):
        page_id = self.create_page()
        self.add_exp_details(page_id)
        self.file_uploader.upload()
        self.file_uploader.attach_to_page(page_id=page_id, prefix_name=self.exp_name)

    def create_page(self):
        response = self.notion.pages.create(
            parent={"database_id": self.database_id},
            properties={
                "Model": {
                    "title": [
                        {
                            "type": "text",
                            "text": {"content": self.exp_name}
                        }
                    ]
                },
                "mAP50": {"number": float(quantize_to_2dp(self.metrics_dict['mAP50']*100))},
                "mAP75": {"number": float(quantize_to_2dp(self.metrics_dict['mAP75']*100))},
                "mAP": {"number": float(quantize_to_2dp(self.metrics_dict['mAP']*100))},
                "Last updated time": {"date": {"start": datetime.datetime.now().isoformat()}},
                "Category": {
                    "multi_select": [{"name": tag} for tag in self.tags]
                },
            }
        )
        page_id = response["id"]  # get page id
        Logger.info(f"✅ exp page added successful：{response['url']}")
        return page_id

    def add_exp_details(self, page_id):
        self.notion.blocks.children.append(
            block_id=page_id,
            children=[build_table_block(self.metrics_dict['metrics_table'])]
        )
        self.notion.blocks.children.append(
            block_id=page_id,
            children=[
                {
                    "object": "block",
                    "type": "code",
                    "code": {
                        "language": "yaml",
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": self.metrics_dict['sys_info']
                                }
                            }
                        ]
                    }
                }
            ]
        )
        Logger.info("✅ details added successful")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload experiment results.')
    parser.add_argument('--exp-path', type=str, default='runs/detect/visdrone/v12s', help='Path to experiment directory')
    parser.add_argument('--data-path', type=str, default='ultralytics/cfg/datasets/VisDrone.yaml', help='Path to dataset configuration file')
    parser.add_argument('--extra-tags', nargs='+', default=['v12', 'v12s'], help='Extra tags for the upload (space-separated)')
    args = parser.parse_args()
    uploader = ExpUpLoader(
        exp_path=args.exp_path,
        data_path=args.data_path,
        extra_tags=args.extra_tags,
    )
    try:
        uploader.upload()
    except Exception as e:
        Logger.error('❌ Upload FAIL')
        Logger.error(traceback.format_exc())

