# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import copy
import glob
import json
from functools import partial
import gradio as gr
from threading import Thread
from time import time
import mlflow
import re

from ralle.evaluation import normalize_answer
from ralle.utils import read_config, interpret_prompt, exec_function, load_llms, load_retrievers
from ralle.evaluation.evaluate import evaluate_ralle

def call_evaluate_ralle(run_name, config_exp=None, config_base=None, funcs=None):
    with mlflow.start_run(run_name=run_name):
        evaluate_ralle(config_exp, config_base, funcs=funcs)

def run_chat(args):

    config_exp, config_base = read_config(args)

    # setup instances
    funcs = {}
    qa_dataset = {}
    questions_show = {}
    used_llm_models = set()
    loaded_llm_models = set()
    used_retriever = set()
    loaded_retriever = set()

    cwd = os.getcwd()
    os.chdir(args.config_exp_path)
    config_files = glob.glob("./**/*.json", recursive=True)
    os.chdir(cwd)
    config_files = sorted(config_files)

    with gr.Blocks(title="RALLE") as app:

        # --- variable ---
        max_len_chain = config_base['system_config']['max_len_chain']

        with gr.Tab("Load models"):

            # --- layout ---
            load_models_status_md = gr.Markdown('Status: ')
            load_llm_ddn = gr.Dropdown([m for m in config_base['llm_config'].keys()], label='LLMs', multiselect=True)
            load_retriever_ddn = gr.Dropdown([m for m in config_base['retriever_config'].keys()], label='Retrievers', multiselect=True)
            load_execute_btn = gr.Button('Load models')

            # --- local (in Load models tab) functions ---
            def status_loading_llm():
                return gr.Markdown.update('Status: loading LLMs...')

            def status_loading_retrievers(loaded_llm=None):
                return gr.Markdown.update('Status: loaded LLMs: **{}**, loading Retrievers...'.format(loaded_llm))

            def status_loading_completed(loaded_llm=None, loaded_ret=None):
                return gr.Markdown.update('Status: loaded LLMs: **{}**, loaded Retrievers: **{}**.'.format(loaded_llm, loaded_ret))

            def select_for_use(selected, used=None):
                used.clear()
                used |= set(selected)
                # print('selected: {}, used: {}'.format(selected, used))
                if len(selected) != 0:
                    out = []
                    for v in config_exp['chain_config']['chain']:
                        if v['function'] == 'Retriever':
                            out.append(gr.Dropdown.update(choices=selected, value=v['retriever_name']))
                        else:
                            out.append(gr.Dropdown.update(choices=selected, value=v['llm_name']))
                    return out + [gr.Dropdown.update(choices=selected, value=selected[0])] * (max_len_chain - len(config_exp['chain_config']['chain']))
                else:
                    return [gr.Dropdown.update(choices=selected, value='')] * max_len_chain

            # --- local (in Develop chain tab) events ---
            load_execute_btn.click(fn=status_loading_llm, inputs=None, outputs=load_models_status_md).then(
                                   fn=partial(load_llms, config_base=config_base, funcs=funcs, loaded_llm=loaded_llm_models), inputs=[load_llm_ddn]).then(
                                   fn=partial(status_loading_retrievers, loaded_llm=loaded_llm_models), inputs=None, outputs=load_models_status_md).then(
                                   fn=partial(load_retrievers, config_base=config_base, funcs=funcs, loaded_ret=loaded_retriever), inputs=[load_retriever_ddn]).then(
                                   fn=partial(status_loading_completed, loaded_llm=loaded_llm_models, loaded_ret=loaded_retriever), inputs=None, outputs=load_models_status_md)

        with gr.Tab("Develop chain"):

            # --- layout ---
            chain_dataset_ddn = gr.Dropdown([m for m in config_base['dataset_config']['develop'].keys()], label='Dataset')
            chain_question_dd = gr.Dropdown(questions_show, value='', type="value", label="Question", allow_custom_value=True)

            with gr.Row():
                chain_load_config_ddn = gr.Dropdown(config_files, label='Config')
                chain_load_config_btn = gr.Button('Load config')
                chain_len_chain_dd = gr.Dropdown([str(i) for i in range(1, max_len_chain + 1)], value=str(config_exp['chain_config']['len_chain']), type="value", label="Chain length")
                chain_execute_chain_btn = gr.Button("Execute entire chain")

            chain_line_mds = []
            chain_label_mds = []
            chain_execute_btns = []
            chain_interpret_prompt_btns = []
            chain_prompt_template_tas = []
            chain_prompt_tas = []
            chain_response_tas = []
            chain_response_hts = []
            chain_response_radios = []
            chain_function_radios = []
            chain_llm_ddns = []
            chain_retriever_ddns = []
            chain_k_tbs = []
            chain_format_prompt_radios = []
            chain_format_prompt_tbs = []
            chain_retrieved_wiki_id_tbs = []
            chain_retrieved_wiki_id_hts = []
            chain_history = {}
            chain_history['prompt'] = []
            chain_history['response'] = []
            chain_history['wiki_id_title'] = []

            chain_rows = []

            for i in range(max_len_chain):
                vis = i < config_exp['chain_config']['len_chain']
                if vis:
                    chain_history['prompt'].append('')
                    chain_history['response'].append('')
                    chain_history['wiki_id_title'].append('')

                temp_val = config_exp['chain_config']['chain'][i]['prompt_template'] if vis else ''
                func = config_exp['chain_config']['chain'][i]['function'] if vis else 'Identity'
                llm = config_exp['chain_config']['chain'][i]['llm_name'] if vis and func=='LLM' else ''
                ret = config_exp['chain_config']['chain'][i]['retriever_name'] if vis and func=='Retriever' else ''
                k_val = str(config_exp['chain_config']['chain'][i]['npassage']) if vis and func=='Retriever' else '0'
                fmt_val = config_exp['chain_config']['chain'][i]['f-strings_or_eval'] if vis else 'f-strings'

                with gr.Column(visible=vis) as r:
                    chain_line_mds.append(gr.Markdown('''---'''))
                    with gr.Row():
                        chain_label_mds.append(gr.Markdown('''## Action {}'''.format(i+1)))
                        chain_interpret_prompt_btns.append(gr.Button("Interpret prompt only"))
                        chain_execute_btns.append(gr.Button("Interpret prompt and execute this action"))
                    with gr.Row():
                        with gr.Column():
                            chain_prompt_template_tas.append(gr.TextArea(label="prompt_template[{}]".format(i), value=temp_val))
                            with gr.Row():
                                chain_format_prompt_radios.append(gr.Radio(['f-strings', 'eval'], value=fmt_val, type='value', label='f-strings or eval'))
                                chain_format_prompt_tbs.append(gr.Textbox("{question}, {response[0]}", label='Example'))
                        with gr.Column():
                            chain_prompt_tas.append(gr.TextArea(label="prompt[{}]".format(i)))
                            chain_function_radios.append(gr.Radio(['LLM', 'Retriever', 'Identity'], value=func, label='Function'))
                            with gr.Row():
                                chain_llm_ddns.append(gr.Dropdown(choices=used_llm_models, value=llm, type='value', label='LLM name', visible=func=='LLM'))
                                chain_retriever_ddns.append(gr.Dropdown(choices=used_retriever, value=ret, type='value', label='Retriever name', visible=func=='Retriever'))
                                chain_k_tbs.append(gr.Textbox(label='k, press ENTER key to apply', value=k_val, visible=func=='Retriever'))
                        with gr.Column():
                            chain_response_tas.append(gr.TextArea(label="response[{}]".format(i)))
                            chain_response_hts.append(gr.HighlightedText(label="response[{}]".format(i), value="", visible=False, show_legend=False))
                            chain_retrieved_wiki_id_tbs.append(gr.Textbox(label='wiki_id_title[{}]'.format(i), value="", visible=func=='Retriever'))
                            chain_retrieved_wiki_id_hts.append(gr.HighlightedText(label='wiki_id_title[{}]'.format(i), value="", visible=False, show_legend=False))
                            chain_response_radios.append(gr.Radio(['plain text', 'highlighted text'], value='plain text', type='value', label='plain or highlighted'))
                chain_rows.append(r)

            with gr.Accordion(label="show qa data", open=False):
                chain_qa_data_json = gr.JSON([qa_dataset[i] for i in range(len(qa_dataset))], label="qa_data")

            # --- local (in Develop chain tab) functions ---

            def select_dataset(dataset, config_exp=None):

                qa_path = config_base['dataset_config']['develop'][dataset]

                t0 = time(); print('loading dataset...')
                with open(qa_path, "r") as fin:
                    for i, line in enumerate(fin):
                        qa_dataset[i] = json.loads(line)
                        if i == config_base['system_config']['num_show_question']:
                            break
                t1 = time(); print('loading dataset... ({:.2f} sec)'.format(t1-t0))

                questions_show.update({i: q['input'].replace('\n', ' ').replace('\"', '') for i, q in qa_dataset.items()})

                config_exp['chain_config']['dataset']['dataset_name'] = dataset

                return [gr.Dropdown.update(choices=[questions_show[i] for i in range(len(questions_show))], value=questions_show[0])] + [gr.JSON.update(value=qa_dataset[0])] + [gr.Json.update(value=config_exp)]

            def update_len_chain(len_chain, config_exp=None):
                len_chain = int(len_chain)
                config_exp['chain_config']['len_chain'] = len_chain
                func = []
                temp = []
                # change config_exp
                while len(config_exp['chain_config']['chain']) < len_chain:
                    config_exp['chain_config']['chain'].append({'prompt_template': '', 'function': 'Identity'})

                # change gui data 
                for i in range(max_len_chain):
                    vis = i < len_chain
                    if vis and len(config_exp['chain_config']['chain']) > i:
                        temp.append(config_exp['chain_config']['chain'][i]['prompt_template'])
                        func.append(config_exp['chain_config']['chain'][i]['function'])
                    else:
                        temp.append('')
                        func.append('Identity')
                
                return [gr.Row.update(visible=True)] * len_chain + [gr.Row.update(visible=False)] * (max_len_chain - len_chain) + func + temp

            def select_question(question, history=None):

                history['question'] = question
                for i, q in questions_show.items():
                    if question == q:
                        return gr.JSON.update(value=qa_dataset[i])
                
                return gr.JSON.update(value={"Not found in dataset."})
            
            def disable():
                return [gr.Button.update(interactive=False)] * (1 + max_len_chain)

            def enable():
                return [gr.Button.update(interactive=True)] * (1 + max_len_chain)

            def action(function, llm, retriever, k, format_prompt, prompt_template, step=None, history=None):
                t_in, t_out, wiki_ids = exec_function(function, llm, retriever, k, format_prompt, prompt_template, step=step, history=history, funcs=funcs)
                return t_in, gr.TextArea.update(t_out, visible=True), gr.TextArea.update(wiki_ids, visible=function=='Retriever'), gr.HighlightedText.update(visible=False, value=None), gr.HighlightedText.update(visible=False, value=None), gr.HighlightedText.update(value='plain text'), gr.Button.update(interactive=True)

            def select_function(func):
                if func == 'Retriever':
                    if len(used_retriever) != 0:
                        return gr.Dropdown.update(visible=False), gr.Dropdown.update(visible=True, choices=used_retriever, value=list(used_retriever)[0]), gr.Textbox.update(visible=True), gr.Textbox.update(visible=True)
                    else:
                        return gr.Dropdown.update(visible=False), gr.Dropdown.update(visible=True, choices=[], value=''), gr.Textbox.update(visible=True), gr.Textbox.update(visible=True)
                elif func == 'LLM':
                    if len(used_llm_models) != 0:
                        return gr.Dropdown.update(visible=True, choices=used_llm_models, value=list(used_llm_models)[0]), gr.Dropdown.update(visible=False), gr.Textbox.update(visible=False), gr.Textbox.update(visible=False)
                    else:
                        return gr.Dropdown.update(visible=True, choices=[], value=''), gr.Dropdown.update(visible=False), gr.Textbox.update(visible=False), gr.Textbox.update(visible=False)
                elif func == 'Identity':
                    return gr.Dropdown.update(visible=False), gr.Dropdown.update(visible=False), gr.Textbox.update(visible=False), gr.Textbox.update(visible=False)

            def check_response(ta_or_ht, question, response, response_ht, wiki_id, wiki_id_ht, chain_id=0):

                if chain_id < config_exp['chain_config']['len_chain']:
                    is_r = config_exp['chain_config']['chain'][chain_id]['function'] == 'Retriever'
                else:
                    is_r = False

                if ta_or_ht == 'highlighted text' and response_ht is None:
                    for i, q in questions_show.items():
                        if question == q:
                            qa = qa_dataset[i]
                            break

                    out_diff_wiki_id = {}
                    out_diff_wiki_id['text'] = wiki_id
                    out_diff_wiki_id['entities'] = []

                    seen_provs = set()
                    idx = 0
                    for ans in qa['output']:
                        if 'provenance' in ans:
                            for prov in ans['provenance']:
                                wiki_id_gt = prov['wikipedia_id']
                                if wiki_id_gt not in seen_provs:
                                    seen_provs.add(wiki_id_gt)
                                    res_iter_gen = re.finditer(wiki_id_gt, wiki_id)
                                    found = False
                                    for res in res_iter_gen:
                                        out_diff_wiki_id['entities'].append({'entity': str(idx), 'word': wiki_id_gt, 'start': res.start(), 'end': res.end()})
                                        found = True
                                    if not found:
                                        out_diff_wiki_id['entities'].append({'entity': str(idx), 'word': wiki_id_gt, 'start': len(wiki_id), 'end': len(wiki_id)})
                                    idx += 1
                    
                    out_wiki_id = [gr.Textbox.update(visible=False), gr.HighlightedText.update(visible=is_r, value=out_diff_wiki_id)]

                    out_diff_gen = {}
                    norm_response = normalize_answer(response)
                    out_diff_gen['text'] = norm_response
                    out_diff_gen['entities'] = []

                    seen_anss = set()
                    idx = 0
                    for anss in qa['output']:
                        if 'answer' in anss:
                            ans = anss['answer']
                            norm_answer = normalize_answer(ans)
                            if norm_answer not in seen_anss:
                                seen_anss.add(norm_answer)
                                res_iter_gen = re.finditer(norm_answer, norm_response)
                                found = False
                                for res in res_iter_gen:
                                    out_diff_gen['entities'].append({'entity': str(idx), 'word': norm_answer, 'start': res.start(), 'end': res.end()})
                                    found = True
                                if not found:
                                    out_diff_gen['entities'].append({'entity': str(idx), 'word': norm_answer, 'start': len(norm_response), 'end': len(norm_response)})
                            idx += 1

                    out_gen = [gr.Textbox.update(visible=False), gr.HighlightedText.update(visible=True, value=out_diff_gen)]

                elif ta_or_ht == 'highlighted text':
                    out_wiki_id = [gr.Textbox.update(visible=False), gr.HighlightedText.update(visible=is_r, value=wiki_id_ht)]
                    out_gen = [gr.Textbox.update(visible=False), gr.HighlightedText.update(visible=True, value=response_ht)]
                else:
                    out_wiki_id = [gr.Textbox.update(visible=is_r), gr.HighlightedText.update(visible=False)]
                    out_gen = [gr.Textbox.update(visible=True), gr.HighlightedText.update(visible=False)]

                return out_gen + out_wiki_id

            def load_config(cfg_file, config_exp=None):

                with open(os.path.join('./scripts/configs/experiment_settings', cfg_file), "r", encoding="utf-8") as reader:
                    text = reader.read()
                config_tmp = json.loads(text)
                config_exp['chain_config'] = config_tmp['chain_config']

                format_prompts = []
                prompt_templates = []
                functions = []
                llms = []
                retrievers = []
                ks = []
                len_chain = config_exp['chain_config']['len_chain']
                assert len_chain == len(config_exp['chain_config']['chain'])

                for i, v in enumerate(config_exp['chain_config']['chain']):
                    format_prompts.append(gr.Radio.update(v['f-strings_or_eval']))
                    prompt_templates.append(gr.TextArea.update(v['prompt_template']))
                    functions.append(gr.Radio.update(v['function']))
                    if v['function'] == 'LLM':
                        default_config_llm_name = v['llm_name']
                        dropdown_llm_name = None    
                    
                        if default_config_llm_name in loaded_llm_models or len(loaded_llm_models) == 0:
                            dropdown_llm_name = default_config_llm_name
                        else:
                            dropdown_llm_name = list(loaded_llm_models)[0]
                            config_exp['chain_config']['chain'][i]['llm_name'] = dropdown_llm_name
                            print(f'Warning!, LLM is changed from {default_config_llm_name} to {dropdown_llm_name} because {default_config_llm_name} is not loaded.')

                        llms.append(gr.Dropdown.update(value=dropdown_llm_name, visible=True))
                        retrievers.append(gr.Dropdown.update(visible=False))
                        ks.append(gr.Textbox.update(visible=False))
                    elif v['function'] == 'Retriever':
                        llms.append(gr.Dropdown.update(visible=False))

                        default_config_retriever_name = v['retriever_name']
                        dropdown_retriever_name = None

                        if default_config_retriever_name in loaded_retriever or len(loaded_retriever) == 0:
                            dropdown_retriever_name = default_config_retriever_name
                        else:
                            dropdown_retriever_name = list(loaded_retriever)[0]
                            config_exp['chain_config']['chain'][i]['retriever_name'] = dropdown_retriever_name
                            print(f'Warning!, Retriever is changed from {default_config_retriever_name} to {dropdown_retriever_name} because {default_config_retriever_name} is not loaded.')

                        
                        retrievers.append(gr.Dropdown.update(value=dropdown_retriever_name, visible=True))
                        ks.append(gr.Textbox.update(v['npassage'], visible=True))
                    elif v['function'] == 'Identity':
                        llms.append(gr.Dropdown.update(visible=False))
                        retrievers.append(gr.Dropdown.update(visible=False))
                        ks.append(gr.Textbox.update(visible=False))
                    else:
                        raise NotImplementedError

                return [gr.Dropdown.update(str(len_chain))] \
                       + [gr.Row.update(visible=True)] * len_chain + [gr.Row.update(visible=False)] * (max_len_chain - len_chain) \
                       + format_prompts + [gr.Dropdown.update('f-strings')] * (max_len_chain - len_chain) \
                       + prompt_templates + [gr.TextArea.update('')] * (max_len_chain - len_chain) \
                       + functions + [gr.Dropdown.update('Identity')] * (max_len_chain - len_chain) \
                       + llms + [None] * (max_len_chain - len_chain) \
                       + retrievers + [None] * (max_len_chain - len_chain) \
                       + ks + [None] * (max_len_chain - len_chain)
            
            def reset_history(len_chain, history=None):
                history.pop('prompt', None)
                history.pop('response', None)
                history.pop('wiki_id_title', None)
                history['prompt'] = []
                history['response'] = []
                history['wiki_id_title'] = []
                for _ in range(int(len_chain)):
                    history['prompt'].append('')
                    history['response'].append('')
                    history['wiki_id_title'].append('')
                return

            # --- local (in Develop chain tab) events ---

            chain_load_config_btn.click(fn=partial(load_config, config_exp=config_exp),
                                        inputs=chain_load_config_ddn,
                                        outputs=[chain_len_chain_dd]
                                                + chain_rows
                                                + chain_format_prompt_radios
                                                + chain_prompt_template_tas
                                                + chain_function_radios
                                                + chain_llm_ddns
                                                + chain_retriever_ddns
                                                + chain_k_tbs)

            chain_len_chain_dd.change(fn=partial(update_len_chain, config_exp=config_exp),
                                      inputs=chain_len_chain_dd,
                                      outputs=chain_rows+chain_function_radios+chain_prompt_template_tas).then(
                                      fn=partial(reset_history, history=chain_history),
                                      inputs=chain_len_chain_dd)

            # select llm or retrieve
            for i in range(max_len_chain):
                chain_function_radios[i].change(fn=select_function, inputs=chain_function_radios[i], 
                    outputs=[chain_llm_ddns[i], chain_retriever_ddns[i], chain_k_tbs[i], chain_retrieved_wiki_id_tbs[i]])

            # interpret prompt
            for i in range(max_len_chain):
                chain_interpret_prompt_btns[i].click(fn=partial(interpret_prompt, step=i, history=chain_history),
                                            inputs=[chain_format_prompt_radios[i], chain_prompt_template_tas[i]],
                                            outputs=[chain_prompt_tas[i]])

            # do indivisual process
            for i in range(max_len_chain):
                chain_execute_btns[i].click(fn=disable, outputs=[chain_execute_chain_btn] + chain_execute_btns).then(
                                            fn=partial(action, step=i, history=chain_history), 
                                            inputs=[chain_function_radios[i],
                                                    chain_llm_ddns[i],
                                                    chain_retriever_ddns[i],
                                                    chain_k_tbs[i],
                                                    chain_format_prompt_radios[i],
                                                    chain_prompt_template_tas[i]],
                                            outputs=[chain_prompt_tas[i],
                                                     chain_response_tas[i],
                                                     chain_retrieved_wiki_id_tbs[i],
                                                     chain_response_hts[i],
                                                     chain_retrieved_wiki_id_hts[i],
                                                     chain_response_radios[i]]).then(
                                            fn=enable, outputs=[chain_execute_chain_btn] + chain_execute_btns)
                
            for i in range(max_len_chain):
                chain_response_radios[i].change(fn=partial(check_response, chain_id=i),
                                            inputs=[chain_response_radios[i], chain_question_dd, chain_response_tas[i], chain_response_hts[i], chain_retrieved_wiki_id_tbs[i], chain_retrieved_wiki_id_hts[i]],
                                            outputs=[chain_response_tas[i], chain_response_hts[i], chain_retrieved_wiki_id_tbs[i], chain_retrieved_wiki_id_hts[i]])

            # do chain
            chain_event = chain_execute_chain_btn.click(fn=partial(reset_history, history=chain_history),
                                                        inputs=chain_len_chain_dd).then(
                                                        fn=disable,
                                                        outputs=[chain_execute_chain_btn] + chain_execute_btns)
            for i in range(0, max_len_chain - 1):
                chain_event = chain_event.then(fn=partial(action, step=i, history=chain_history), 
                                               inputs=[chain_function_radios[i],
                                                       chain_llm_ddns[i],
                                                       chain_retriever_ddns[i],
                                                       chain_k_tbs[i],
                                                       chain_format_prompt_radios[i],
                                                       chain_prompt_template_tas[i]],
                                               outputs=[chain_prompt_tas[i],
                                                        chain_response_tas[i],
                                                        chain_retrieved_wiki_id_tbs[i],
                                                        chain_response_hts[i],
                                                        chain_retrieved_wiki_id_hts[i],
                                                        chain_response_radios[i]])
            chain_event = chain_event.then(fn=enable, outputs=[chain_execute_chain_btn] + chain_execute_btns)

            # select / input question
            chain_question_dd.change(fn=partial(select_question, history=chain_history), inputs=chain_question_dd, outputs=chain_qa_data_json)

        with gr.Tab("Chat"):

            # --- layout ---
            chatbot = gr.Chatbot(elem_id="chatbot")
            with gr.Row():
                chat_user_message_tb = gr.Textbox(show_label=False, placeholder="Enter text and press enter", container=False)
                
            # --- local (in Chat tab) functions ---answer
            def user(user_message, history):
                if history is None:
                    history = ""
                return "", history + [[user_message, None]]

            def bot(history):
                chain_history = {}
                reset_history(config_exp['chain_config']['len_chain'], history=chain_history)
                chain_history['question'] = history[-1][0]

                for c in range(config_exp['chain_config']['len_chain']):

                    if config_exp['chain_config']['chain'][c]['prompt_template'] == '':
                        # do nothing
                        break

                    if c != config_exp['chain_config']['len_chain'] -1:
                        t_in, t_out, wiki_ids = exec_function(config_exp['chain_config']['chain'][c]['function'],
                                                              config_exp['chain_config']['chain'][c]['llm_name'] if config_exp['chain_config']['chain'][c]['function']=='LLM' else '',
                                                              config_exp['chain_config']['chain'][c]['retriever_name'] if config_exp['chain_config']['chain'][c]['function']=='Retriever' else '',
                                                              config_exp['chain_config']['chain'][c]['npassage'] if config_exp['chain_config']['chain'][c]['function']=='Retriever' else '',
                                                              config_exp['chain_config']['chain'][c]['f-strings_or_eval'],
                                                              config_exp['chain_config']['chain'][c]['prompt_template'],
                                                              step=c,
                                                              history=chain_history,
                                                              funcs=funcs)
                        
                        history[-1][1] = t_out
                        # print('c: {}, t_in: {}, t_out: {}, wiki_ids: {}'.format(c, t_in, t_out, wiki_ids))

                    else:
                        # for streaming output
                        t_in = interpret_prompt(config_exp['chain_config']['chain'][c]['f-strings_or_eval'],
                                                config_exp['chain_config']['chain'][c]['prompt_template'],
                                                step=c,
                                                history=chain_history)

                        thread = Thread(target=funcs['llms'][config_exp['chain_config']['chain'][c]['llm_name']], kwargs={"prompts": t_in, "streamer": funcs['streamer'][config_exp['chain_config']['chain'][c]['llm_name']]})
                        thread.start()
                        t_out = ''
                        for new_text in funcs['streamer'][config_exp['chain_config']['chain'][c]['llm_name']]:
                            t_out += new_text
                            history[-1][1] = t_out
                            yield history

                yield history
                return

            # --- local (in Chat tab) events ---
            chat_user_message_tb.submit(fn=user, inputs=[chat_user_message_tb, chatbot], outputs=[chat_user_message_tb, chatbot]).then(
                                        fn=bot, inputs=chatbot, outputs=chatbot)

        with gr.Tab("Config"):

            # --- layout ---
            with gr.Row():
                config_do_evaluation_btn = gr.Button("Evaluate")
                config_run_name_tb = gr.Textbox(label='Run name')
                config_num_evaluation_tb = gr.Textbox(label="Number of questions for evaluation, press ENTER key to apply", placeholder="-1 for all")
                config_batch_size_tb = gr.Textbox(label="Batch size, press ENTER key to apply")
            with gr.Row():
                config_refresh_btn = gr.Button("Refresh configuration")
            with gr.Row():
                config_json = gr.JSON(config_exp, label="Current config")
            with gr.Row():
                config_save_path = gr.Textbox(label='Save path', value='scripts/configs/experiment_settings/tmp_new.json')
                config_save_btn = gr.Button("Save config")

            # --- local (in Config tab) functions ---

            def save_config(path):
                kwargs_show = copy.deepcopy(config_exp)
                kwargs_show['chain_config']['chain'] = []
                for i in range(config_exp['chain_config']['len_chain']):
                    kwargs_show['chain_config']['chain'].append(config_exp['chain_config']['chain'][i])

                with open(path, "w", encoding="utf-8") as writer:
                    json.dump(kwargs_show, writer, ensure_ascii=False, indent=4)
                print("Save Config Completed")
                return

            def change_num_eval(num_eval):
                config_exp['chain_config']['dataset']['num_evaluate'] = int(num_eval)

            def change_batch_size(batch_size):
                config_exp['chain_config']['dataset']['batch_size'] = int(batch_size)

            # --- local (in Config tab) events ---

            config_save_btn.click(fn=save_config, inputs=config_save_path)
            config_num_evaluation_tb.submit(fn=change_num_eval, inputs=config_num_evaluation_tb)
            config_batch_size_tb.submit(fn=change_batch_size, inputs=config_batch_size_tb)
            config_do_evaluation_btn.click(fn=partial(call_evaluate_ralle, config_exp=config_exp, config_base=config_base, funcs=funcs), inputs=config_run_name_tb)

        # --- inter-tab functions ---

        # Update config_exp function
        
        def update_config_exp(len_chain, dataset, prompt_template, func, llm, ret, k, fmt, step=None, config_exp=None):
            len_chain = int(len_chain)
            config_exp['chain_config']['len_chain'] = len_chain
            config_exp['chain_config']['dataset']['dataset_name'] = dataset

            if step >= len_chain:
                pass
            else:
                config_exp['chain_config']['chain'][step]['prompt_template'] = prompt_template
                config_exp['chain_config']['chain'][step]['f-strings_or_eval'] = fmt
                config_exp['chain_config']['chain'][step]['function'] = func
                if len(used_llm_models) != 0 and config_exp['chain_config']['chain'][step]['function'] == 'LLM':
                    config_exp['chain_config']['chain'][step]['llm_name'] = llm
                    config_exp['chain_config']['chain'][step].pop('retriever_name', None)
                    config_exp['chain_config']['chain'][step].pop('npassage', None)
                elif len(used_retriever) != 0 and config_exp['chain_config']['chain'][step]['function'] == 'Retriever':
                    config_exp['chain_config']['chain'][step].pop('llm_name', None)
                    config_exp['chain_config']['chain'][step]['retriever_name'] = ret
                    config_exp['chain_config']['chain'][step]['npassage'] = int(k)
                elif config_exp['chain_config']['chain'][step]['function'] == 'Identity':
                    config_exp['chain_config']['chain'][step].pop('llm_name', None)
                    config_exp['chain_config']['chain'][step].pop('retriever_name', None)
                    config_exp['chain_config']['chain'][step].pop('npassage', None)

            return

        def update_config_view():
            return config_exp['chain_config']
        
        def update_format_md(fmt):
            if fmt == 'f-strings':
                return gr.Textbox.update('{question}, {response[0]}')
            elif fmt == 'eval':
                return gr.Textbox.update("\'{}\'.format(question) + \', \' + \'{}\'.format(wiki_id_title[0])")

        # --- inter-tab events ---

        load_llm_ddn.change(fn=partial(select_for_use, used=used_llm_models), inputs=load_llm_ddn, outputs=chain_llm_ddns)
        load_retriever_ddn.change(fn=partial(select_for_use, used=used_retriever), inputs=load_retriever_ddn, outputs=chain_retriever_ddns)
        chain_dataset_ddn.change(fn=partial(select_dataset, config_exp=config_exp), inputs=chain_dataset_ddn, outputs=[chain_question_dd, chain_qa_data_json, config_json])

        for i in range(max_len_chain):
            chain_prompt_template_tas[i].change(
                fn=partial(update_config_exp, config_exp=config_exp, step=i),
                inputs=[chain_len_chain_dd, chain_dataset_ddn, chain_prompt_template_tas[i], chain_function_radios[i], chain_llm_ddns[i], chain_retriever_ddns[i], chain_k_tbs[i], chain_format_prompt_radios[i]]
                )
            chain_function_radios[i].change(
                fn=partial(update_config_exp, config_exp=config_exp, step=i),
                inputs=[chain_len_chain_dd, chain_dataset_ddn, chain_prompt_template_tas[i], chain_function_radios[i], chain_llm_ddns[i], chain_retriever_ddns[i], chain_k_tbs[i], chain_format_prompt_radios[i]]
                )
            chain_llm_ddns[i].change(
                fn=partial(update_config_exp, config_exp=config_exp, step=i),
                inputs=[chain_len_chain_dd, chain_dataset_ddn, chain_prompt_template_tas[i], chain_function_radios[i], chain_llm_ddns[i], chain_retriever_ddns[i], chain_k_tbs[i], chain_format_prompt_radios[i]]
                )
            chain_retriever_ddns[i].change(
                fn=partial(update_config_exp, config_exp=config_exp, step=i),
                inputs=[chain_len_chain_dd, chain_dataset_ddn, chain_prompt_template_tas[i], chain_function_radios[i], chain_llm_ddns[i], chain_retriever_ddns[i], chain_k_tbs[i], chain_format_prompt_radios[i]]
                )
            chain_k_tbs[i].submit(
                fn=partial(update_config_exp, config_exp=config_exp, step=i),
                inputs=[chain_len_chain_dd, chain_dataset_ddn, chain_prompt_template_tas[i], chain_function_radios[i], chain_llm_ddns[i], chain_retriever_ddns[i], chain_k_tbs[i], chain_format_prompt_radios[i]]
                )
            chain_format_prompt_radios[i].change(
                fn=partial(update_config_exp, config_exp=config_exp, step=i),
                inputs=[chain_len_chain_dd, chain_dataset_ddn, chain_prompt_template_tas[i], chain_function_radios[i], chain_llm_ddns[i], chain_retriever_ddns[i], chain_k_tbs[i], chain_format_prompt_radios[i]]
                ).then(
                fn=update_format_md,
                inputs=chain_format_prompt_radios[i],
                outputs=chain_format_prompt_tbs[i]
                )

        config_refresh_btn.click(fn=update_config_view, outputs=config_json)

    app.queue()
    app.launch(share=False, server_name="0.0.0.0")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("run evaluate")
    parser.add_argument("--config", default='scripts/configs/experiment_settings/default.json', help="configuration of retrieval-augmented LLM for evaluateion")
    parser.add_argument("--config_exp_path", default='scripts/configs/experiment_settings', help="path to config_exp directory")
    parser.add_argument("--config_system", default='scripts/configs/base_settings/system.json', help="path to system config file")
    parser.add_argument("--config_llms", default='scripts/configs/base_settings/llms.json', help="path to llm config file")
    parser.add_argument("--config_retrievers", default='scripts/configs/base_settings/retrievers.json', help="path to retriever config file")
    parser.add_argument("--config_query_encoders", default='scripts/configs/base_settings/query_encoders.json', help="path to query encoders config file")
    parser.add_argument("--config_corpora", default='scripts/configs/base_settings/corpora.json', help="path to corpora config file")
    parser.add_argument("--config_datasets", default='scripts/configs/base_settings/datasets.json', help="path to datasets config file")
    args = parser.parse_args()

    run_chat(args)
