# *_*coding:utf-8 *_*
# @Author : YueMengRui
from flask import current_app
from copy import deepcopy
from info import knowledge_vector_store
from info.utils.Prompt_Templates import prompt_templates


class TaskDataHandler:

    def __init__(self):
        self.custom_configs = {}
        self.generation_configs = {}
        self.task_mapping = {
            "sqa": (self._sqa, prompt_templates['sqa']),
            "knowledge": (self._knowledge, prompt_templates['knowledge']),
            "translation": (self._translation, prompt_templates['translation']),
            "summary": (self._summary, prompt_templates['summary'])
        }

        self.reset_configs()

    def reset_configs(self):
        self.generation_configs = {}
        self.generation_configs.update({'max_length': 8192})
        self.custom_configs = {}
        self.generation_configs.update({'max_prompt_length': self.generation_configs['max_length'] - 1000})

    def auto_handler(self, request_datas):
        origin_query_list = []
        prompt_list = []
        history_list = []
        sources = []

        for query_dict in request_datas:
            task_type = query_dict.get('task_type', 'sqa')
            generation_configs = query_dict.get('generation_configs', {})
            prompt = query_dict.get('prompt', None)
            history = query_dict.get('history', [])

            if isinstance(generation_configs, dict):
                self.generation_configs.update(generation_configs)

            if prompt:
                origin_query_list.append(prompt)
                prompt_list.append(prompt)
                history_list.append(history)
                sources.append([])
                continue

            if task_type not in self.task_mapping:
                task_type = 'sqa'

            task_func, base_prompt_template = self.task_mapping[task_type]

            res = task_func(query_dict, base_prompt_template)

            origin_query_list.append(res[0])
            prompt_list.append(res[1])
            history_list.append(res[2])
            sources.append(res[3])

        generation_configs = deepcopy(self.generation_configs)
        custom_configs = deepcopy(self.custom_configs)
        current_app.logger.info(str({'generation_configs': generation_configs}) + '\n')
        current_app.logger.info(str({'custom_configs': custom_configs}) + '\n')
        self.reset_configs()
        return origin_query_list, prompt_list, history_list, sources, generation_configs, custom_configs

    def _sqa(self, data, base_prompt_template):
        query = data.get('query', '')
        history = data.get('history', [])
        file_hashs = data.get('file_hashs', [])
        prompt_template = data.get('prompt_template', None)
        custom_configs = data.get('custom_configs', {})
        if isinstance(custom_configs, dict):
            self.custom_configs.update(custom_configs)

        if not prompt_template or not ('{context}' in prompt_template and '{query}' in prompt_template):
            prompt_template = base_prompt_template

        prompt, related_docs = knowledge_vector_store.generate_knowledge_based_prompt(query.strip(),
                                                                                      file_hashs,
                                                                                      prompt_template=prompt_template,
                                                                                      **self.custom_configs)

        temp = {}
        for doc in related_docs:
            file_hash = doc.metadata['file_hash']
            if file_hash in temp.keys():
                temp[file_hash]['related_content'].append({'context': doc.page_content, 'score': doc.metadata['score']})
            else:
                temp[file_hash] = {'file_hash': file_hash,
                                   'related_content': [{'context': doc.page_content, 'score': doc.metadata['score']}]}

        source = [v for v in temp.values()]

        return query, prompt, history, source

    def _knowledge(self, data, base_prompt_template):
        query = data.get('query', '')
        history = data.get('history', [])
        file_hashs = data.get('file_hashs', [])
        prompt_template = data.get('prompt_template', None)
        custom_configs = data.get('custom_configs', {})

        if isinstance(custom_configs, dict):
            self.custom_configs.update(custom_configs)

        if not prompt_template or not ('{context}' in prompt_template and '{query}' in prompt_template):
            prompt_template = base_prompt_template

        prompt, related_docs = knowledge_vector_store.generate_knowledge_based_prompt(query.strip(),
                                                                                      file_hashs,
                                                                                      prompt_template=prompt_template,
                                                                                      **self.custom_configs)

        temp = {}
        for doc in related_docs:
            file_hash = doc.metadata['file_hash']
            if file_hash in temp.keys():
                temp[file_hash]['related_content'].append({'context': doc.page_content, 'score': doc.metadata['score']})
            else:
                temp[file_hash] = {'file_hash': file_hash,
                                   'related_content': [{'context': doc.page_content, 'score': doc.metadata['score']}]}

        source = [v for v in temp.values()]

        return query, prompt, history, source

    def _translation(self, data, base_prompt_template):
        query = data.get('query', '')
        # history = data.get('history', [])
        prompt_template = data.get('prompt_template', None)

        if not (prompt_template and '{query}' in prompt_template):
            prompt_template = base_prompt_template

        prompt = prompt_template.format(query=query)

        return query, prompt, [], None

    def _summary(self, data, base_prompt_template):
        query = data.get('query', '')
        history = data.get('history', [])
        prompt_template = data.get('prompt_template', None)

        if not (prompt_template and '{query}' in prompt_template):
            prompt_template = base_prompt_template

        prompt = prompt_template.format(query=query)

        return query, prompt, history, None


task_data_handler = TaskDataHandler()
