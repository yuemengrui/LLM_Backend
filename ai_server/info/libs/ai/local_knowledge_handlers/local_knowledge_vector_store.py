# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import numpy as np
from copy import deepcopy
from info.utils.MD5_Utils import md5hex
from typing import List, Tuple
from .file_loader import load_file
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


def seperate_list(ls: List[int], index_to_docstore_id, docstore) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    file_hash = docstore.search(index_to_docstore_id[ls1[-1]]).metadata['file_hash']
    for i in range(1, len(ls)):
        _file_hash = docstore.search(index_to_docstore_id[ls[i]]).metadata['file_hash']
        if _file_hash == file_hash:
            ls1.append(ls[i])
        else:
            lists.append(deepcopy(ls1))
            ls1 = [ls[i]]
            file_hash = _file_hash

    lists.append(ls1)
    return lists


def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, knowledge_chunk_size=512, knowledge_score_threshold=250,
        knowledge_good_score=50, logger=None, **kwargs
) -> List[Tuple[Document, float]]:
    scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
    docs = []
    id_set = set()
    store_len = len(self.index_to_docstore_id)
    for j, i in enumerate(indices[0]):
        if i == -1:
            # This happens when not enough docs are returned.
            continue
        _id = self.index_to_docstore_id[i]
        doc = self.docstore.search(_id)
        if not self.chunk_conent:
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc.metadata["score"] = int(scores[0][j])
            docs.append(doc)
            continue
        id_set.add(i)
        docs_len = len(doc.page_content)
        chunk_size = knowledge_chunk_size * (
                (knowledge_score_threshold - max(knowledge_good_score, int(scores[0][j]))) / (
                knowledge_score_threshold - knowledge_good_score))

        if logger:
            logger.info(str({'knowledge_chunk_size': knowledge_chunk_size,
                             'knowledge_score_threshold': knowledge_score_threshold,
                             'knowledge_good_score': knowledge_good_score, 'chunk_size': chunk_size}) + '\n')

        for k in range(1, max(i, store_len - i)):
            break_flag = False
            for l in [i + k, i - k]:
                if 0 <= l < len(self.index_to_docstore_id):
                    _id0 = self.index_to_docstore_id[l]
                    doc0 = self.docstore.search(_id0)
                    if docs_len + len(doc0.page_content) > chunk_size:
                        break_flag = True
                        break
                    elif doc0.metadata["source"] == doc.metadata["source"]:
                        docs_len += len(doc0.page_content)
                        id_set.add(l)
            if break_flag:
                break
    if not self.chunk_conent:
        return docs
    if len(id_set) == 0:
        return []
    id_list = sorted(list(id_set))
    id_lists = seperate_list(id_list, self.index_to_docstore_id, self.docstore)
    for id_seq in id_lists:
        for id in id_seq:
            if id == id_seq[0]:
                _id = self.index_to_docstore_id[id]
                doc = self.docstore.search(_id)
            else:
                _id0 = self.index_to_docstore_id[id]
                doc0 = self.docstore.search(_id0)
                doc.page_content += " " + doc0.page_content
        if not isinstance(doc, Document):
            raise ValueError(f"Could not find document for id {_id}, got {doc}")
        doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
        doc.metadata["score"] = int(doc_score)
        docs.append(doc)

    return docs


class KnowledgeVectorStore:

    def __init__(self,
                 vector_store_root_dir,
                 embedding_model_name_or_path,
                 embedding_device='cuda',
                 init_knowledge_dir='./Init_Knowledges',
                 logger=None
                 ):
        self.vector_store_root_dir = vector_store_root_dir

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name_or_path,
                                                model_kwargs={'device': embedding_device})

        self.logger = logger
        self.init_knowledge_dir = init_knowledge_dir
        self.init_knowledges = []
        self.init_knowledge()

    def init_knowledge(self):
        self.init_knowledges = []
        if os.path.exists(self.init_knowledge_dir):
            file_list = [os.path.join(self.init_knowledge_dir, x) for x in os.listdir(self.init_knowledge_dir)]
            for file_path in file_list:
                with open(file_path, 'rb') as f:
                    file_data = f.read()

                file_hash = md5hex(file_data)

                if file_hash:
                    if self.build_vector_store(file_path, file_hash):
                        self.init_knowledges.append(os.path.join(self.vector_store_root_dir, file_hash))
                        self.write_log({'init_knowledge': file_hash + '--' + file_path})

    def write_log(self, msg):
        if self.logger:
            self.logger.info(str(msg) + '\n')
        else:
            print(msg)

    def check_vector_exist(self, file_hash):
        if os.path.exists(os.path.join(self.vector_store_root_dir, file_hash)):
            return True

        return False

    def check_vector_store(self, vector_list):
        true_vector_list = []

        for i in vector_list:
            vector_dir = os.path.join(self.vector_store_root_dir, i)
            if os.path.exists(vector_dir):
                true_vector_list.append(vector_dir)

        return list(set(true_vector_list))

    def build_vector_store(self, filepath, file_hash):
        if os.path.exists(os.path.join(self.vector_store_root_dir, file_hash)):
            self.write_log({"file hash exist": file_hash})
            return True
        loaded_files = []
        docs = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                self.write_log({"file load error": "路径不存在"})
                return False
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_file(filepath)
                    self.write_log({"file load": "{}已成功加载".format(file)})
                    loaded_files.append(filepath)
                except Exception as e:
                    self.write_log({'file load error': '{}未能成功加载: {}'.format(file, str(e))})
                    return False
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        docs += load_file(fullfilepath)
                        self.write_log({"file load": "{}已成功加载".format(file)})
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        self.write_log({'file load error': '{}未能成功加载: {}'.format(file, str(e))})
        else:
            for file in filepath:
                try:
                    docs += load_file(file)
                    self.write_log({"file load": "{}已成功加载".format(file)})
                    loaded_files.append(file)
                except Exception as e:
                    self.write_log({'file load error': '{}未能成功加载: {}'.format(file, str(e))})

        if len(docs) > 0:
            for doc in docs:
                doc.metadata.update({'file_hash': file_hash})
            self.write_log({'load_doc': docs})
            vector_store_dir = os.path.join(self.vector_store_root_dir, file_hash)
            vector_store = FAISS.from_documents(docs, self.embeddings)

            vector_store.save_local(vector_store_dir)
            # self.vector_store_dir_list.append(file_hash)
            return True
        else:
            self.write_log({'file load error': '文件均未能成功加载'})
            return False

    def get_docs_with_score(self, docs_with_score, knowledge_score_rate=0, knowledge_score_threshold=200, **kwargs):
        docs_with_score.sort(key=lambda x: x.metadata['score'])

        self.write_log({'related_docs_with_score': docs_with_score})
        docs = []
        others_list = []

        if not isinstance(knowledge_score_threshold, int):
            knowledge_score_threshold = 200

        self.write_log(
            {'knowledge_score_rate': knowledge_score_rate, 'knowledge_score_threshold': knowledge_score_threshold})
        for doc in docs_with_score:
            if doc.metadata['score'] < knowledge_score_threshold:
                docs.append(doc)
            else:
                others_list.append(doc)

        if isinstance(knowledge_score_rate, (float, int)) and knowledge_score_rate > 0 and docs:
            first_score = docs[0].metadata['score']
            up_score = first_score + first_score * knowledge_score_rate
            for i in docs[::-1]:
                if i.metadata['score'] > up_score:
                    docs.remove(i)

        return docs

    def get_related_docs(self, query, vector_store_dir_list, knowledge_chunk_size=512, knowledge_chunk_connect=True,
                         vector_search_top_k=10, **kwargs):
        if self.init_knowledges:
            vector_store_dir_list.extend(self.init_knowledges)

        if len(vector_store_dir_list) == 0:
            return []

        for i, vector_store_dir in enumerate(vector_store_dir_list):
            if i == 0:
                vector_store = FAISS.load_local(vector_store_dir, self.embeddings)
            else:
                vector_store.merge_from(FAISS.load_local(vector_store_dir, self.embeddings))

        if not isinstance(knowledge_chunk_size, (int, float)):
            knowledge_chunk_size = 512

        if not isinstance(knowledge_chunk_connect, bool):
            knowledge_chunk_connect = True

        if not isinstance(vector_search_top_k, int):
            vector_search_top_k = 10

        self.write_log(
            {'knowledge_chunk_size': knowledge_chunk_size, 'knowledge_chunk_connect': knowledge_chunk_connect,
             'vector_search_top_k': vector_search_top_k})
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_conent = knowledge_chunk_connect

        related_docs_with_score = vector_store.similarity_search_with_score(query, k=vector_search_top_k,
                                                                            logger=self.logger, **kwargs)

        related_docs = self.get_docs_with_score(related_docs_with_score, **kwargs)

        return related_docs

    def generate_prompt(self, related_docs: List[str],
                        query: str,
                        max_prompt_len: int,
                        prompt_template=None) -> str:
        if not related_docs:
            return query, []

        self.write_log({'related_docs': related_docs})
        base_prompt_len = len(prompt_template.format(context='', query=query))
        true_related_docs = []

        for i in related_docs:
            if base_prompt_len + len(i.page_content) > max_prompt_len:
                break

            true_related_docs.append(i)
            base_prompt_len += len(i.page_content)

        if not true_related_docs:
            return query, []

        context = "\n".join([doc.page_content for doc in true_related_docs])
        self.write_log({'context_len': len(context), 'context': context})
        prompt = prompt_template.format(context=context, query=query)

        return prompt, true_related_docs

    def generate_knowledge_based_prompt(self, query, vector_store_dir_list, max_prompt_length=3096,
                                        prompt_template=None,
                                        **kwargs):
        self.write_log({'max_prompt_length': max_prompt_length, 'prompt_template': prompt_template})
        self.write_log(kwargs)
        vector_store_dir_list = self.check_vector_store(vector_store_dir_list)
        related_docs = self.get_related_docs(query, vector_store_dir_list, **kwargs)

        knowledge_based_prompt, docs = self.generate_prompt(related_docs, query, max_prompt_length, prompt_template)

        return knowledge_based_prompt, docs
