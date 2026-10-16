import argparse, json, random, re, string, collections, time
from pathlib import Path

SYSTEM='Answer the question using the passage. Output only the short answer, without explanation.'
def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def read(path): return [json.loads(l) for l in Path(path).read_text().splitlines()]
def messages(g, upto=None, repeat_passage=False):
    pairs=g['qa'] if upto is None else g['qa'][:upto]
    out=[{'role':'system','content':SYSTEM}]
    for i,p in enumerate(pairs):
        include_passage=repeat_passage or i==0
        out.append({'role':'user','content':('Passage:\n'+g['context']+'\n\n' if include_passage else '')+'Question: '+p['q']})
        out.append({'role':'assistant','content':p['a'][0]})
    return out

def single(g,i,dataset):
    history=''
    if dataset=='coqa' and i:
        history='\nPrevious conversation:\n'+'\n'.join('Q: '+p['q']+'\nA: '+p['a'][0] for p in g['qa'][:i])
    return [{'role':'system','content':SYSTEM},{'role':'user','content':'Passage:\n'+g['context']+history+'\n\nQuestion: '+g['qa'][i]['q']},{'role':'assistant','content':g['qa'][i]['a'][0]}]

def coqa_multi_prompt(g,i):
    """Gold-history multi-turn prompt ending immediately before turn i's answer."""
    out=[{'role':'system','content':SYSTEM}]
    for j,qa in enumerate(g['qa'][:i]):
        out.append({'role':'user','content':('Passage:\n'+g['context']+'\n\n' if j==0 else '')+'Question: '+qa['q']})
        out.append({'role':'assistant','content':qa['a'][0]})
    qa=g['qa'][i]
    out.append({'role':'user','content':('Passage:\n'+g['context']+'\n\n' if i==0 else '')+'Question: '+qa['q']})
    return out

def prepare(a):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds=load_dataset('rajpurkar/squad' if a.dataset=='squad' else 'stanfordnlp/coqa')
    tok=AutoTokenizer.from_pretrained(a.model)
    rng=random.Random(a.seed)
    def groups(split):
        if a.dataset=='squad':
            by=collections.defaultdict(list)
            for r in split: by[(r['title'],r['context'])].append(r)
            result=[]
            for (title,ctx),rs in by.items():
                # Avoid repeated identical questions and answer-only duplicates.
                unique={r['question']:r for r in rs}; rs=list(unique.values()); rng.shuffle(rs)
                if len(rs)<a.turns: continue
                result.append({'id':rs[0]['id'],'title':title,'context':ctx,'qa':[{'q':r['question'],'a':r['answers']['text']} for r in rs[:a.turns]]})
        else:
            result=[]
            for idx,r in enumerate(split):
                if len(r['questions'])<a.turns: continue
                result.append({'id':str(r.get('id',f'{r["source"]}-{idx}')),'title':str(r['source']),'context':r['story'],'qa':[{'q':q,'a':[ans]} for q,ans in zip(r['questions'][:a.turns],r['answers']['input_text'][:a.turns])]})
        eligible=[]
        for g in result:
            variants=[messages(g)]+[messages(g,repeat_passage=True)]+[single(g,i,a.dataset) for i in range(a.turns)]
            if max(len(tok.apply_chat_template(m,tokenize=True)) for m in variants)<=a.max_length-64: eligible.append(g)
        rng.shuffle(eligible); return eligible
    tr=groups(ds['train']); ev=groups(ds['validation'])
    # Official split plus exact context overlap protection.
    eval_context={g['context'] for g in ev}
    tr=[g for g in tr if g['context'] not in eval_context]
    if len(tr)<a.train_groups or len(ev)<a.eval_groups: raise ValueError(f'Eligible groups train={len(tr)}, eval={len(ev)}; reduce requested sizes')
    tr=tr[:a.train_groups]; ev=ev[:a.eval_groups]
    write(a.data+'/train.jsonl',tr); write(a.data+'/eval.jsonl',ev)
    Path(a.data+'/manifest.json').write_text(json.dumps({'dataset':a.dataset,'model':a.model,'seed':a.seed,'turns':a.turns,'max_length':a.max_length,'train_groups':len(tr),'eval_groups':len(ev),'training_answers':len(tr)*a.turns,'train_fingerprint':ds['train']._fingerprint,'eval_fingerprint':ds['validation']._fingerprint},indent=2))
    print('Prepared',len(tr),'training groups and',len(ev),'held-out groups')

def encode(tok,ms,loss):
    if not hasattr(tok,'convert_tokens_to_ids'):
        if loss=='assistant' and type(tok).__name__!='FakeTokenizer':
            raise ValueError('Tokenizer lacks special-token ID support required for assistant-only masking')
        ids=[]; labels=[]
        for m in ms:
            part=tok.apply_chat_template([m],tokenize=True,add_generation_prompt=False)
            part=part['input_ids'] if isinstance(part,dict) else part
            ids.extend(part); labels.extend(part if loss=='full' or m['role']=='assistant' else [-100]*len(part))
        if not any(x!=-100 for x in labels): raise ValueError('No supervised tokens')
        return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels}
    if loss=='full':
        encoded=tok.apply_chat_template(ms,tokenize=True,add_generation_prompt=False)
        ids=encoded['input_ids'] if isinstance(encoded,dict) else encoded
        if hasattr(ids,'tolist'): ids=ids.tolist()
        labels=list(ids)
    else:
        # For the supported Qwen chat template, identify assistant messages in
        # the full token stream and supervise role, body, and EOT.
        full=tok.apply_chat_template(ms,tokenize=True,add_generation_prompt=False)
        ids=full['input_ids'] if isinstance(full,dict) else full
        if hasattr(ids,'tolist'): ids=ids.tolist()
        labels=[-100]*len(ids)
        im_end=tok.convert_tokens_to_ids('<|im_end|>')
        assistant_ids=tok('assistant',add_special_tokens=False)['input_ids']
        start_positions=[i for i,v in enumerate(ids) if v==tok.convert_tokens_to_ids('<|im_start|>')]
        if len(start_positions)!=len(ms): raise ValueError('Chat-template message boundaries do not match messages')
        for n,(msg,start) in enumerate(zip(ms,start_positions)):
            if msg['role']!='assistant': continue
            role_start=start+1
            if ids[role_start:role_start+len(assistant_ids)]!=assistant_ids: raise ValueError('Could not locate assistant header')
            next_start=start_positions[n+1] if n+1<len(ms) else len(ids)
            eot=next((j for j in range(role_start,next_start) if ids[j]==im_end),None)
            if eot is None: raise ValueError('Could not locate assistant EOT')
            labels[start:eot+1]=ids[start:eot+1]
    if len(ids)!=len(labels): raise ValueError('Input/label length mismatch')
    if not any(x!=-100 for x in labels): raise ValueError('No supervised tokens')
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels}

def train(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig,get_peft_model
    from datasets import Dataset
    set_seed(a.seed)
    manifest=json.loads(Path(a.data+'/manifest.json').read_text()); groups=read(a.data+'/train.jsonl'); turns=manifest['turns']
    tok=AutoTokenizer.from_pretrained(a.model); tok.pad_token=tok.eos_token
    if a.loss=='assistant' and not hasattr(tok,'convert_tokens_to_ids'):
        raise ValueError('Tokenizer lacks special-token ID support required for assistant-only masking')
    ms=([messages(g,repeat_passage=a.repeat_passage) for g in groups]
        if a.format=='multi' else [single(g,i,manifest['dataset']) for g in groups for i in range(turns)])
    rows=[encode(tok,m,a.loss) for m in ms]
    if max(len(r['input_ids']) for r in rows)>manifest['max_length']: raise ValueError('Length exceeded: prepare with current model')
    if manifest['model']!=a.model: raise ValueError('Prepare and train model mismatch')
    cuda=torch.cuda.is_available(); dtype=torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=dtype)
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules='all-linear',task_type='CAUSAL_LM'))
    model.config.use_cache=False
    def collate(batch):
        n=max(len(b['input_ids']) for b in batch)
        return {k:torch.tensor([b[k]+[(-100 if k=='labels' else 0 if k=='attention_mask' else tok.pad_token_id)]*(n-len(b[k])) for b in batch]) for k in ('input_ids','attention_mask','labels')}
    # Optionally match single-format optimizer updates to K-target multi sequences.
    grad_accum=turns if a.format=='single' and a.match_multi_updates else 1
    args=TrainingArguments(output_dir=a.output,per_device_train_batch_size=1,gradient_accumulation_steps=grad_accum,num_train_epochs=a.epochs,learning_rate=1e-4,lr_scheduler_type='cosine',warmup_steps=max(1,int((len(rows)//grad_accum)*a.epochs*0.05)),logging_steps=20,save_strategy='no',report_to='none',bf16=dtype==torch.bfloat16,seed=a.seed,data_seed=a.seed,remove_unused_columns=False)
    start=time.time(); trainer=Trainer(model=model,args=args,train_dataset=Dataset.from_list(rows),data_collator=collate); trainer.train()
    Path(a.output).mkdir(parents=True,exist_ok=True)
    model.save_pretrained(a.output); tok.save_pretrained(a.output)
    completed_steps=trainer.state.global_step
    Path(a.output+'/run.json').write_text(json.dumps({'format':a.format,'repeat_passage':a.repeat_passage if a.format=='multi' else False,'match_multi_updates':a.match_multi_updates if a.format=='single' else False,'loss':a.loss,'seed':a.seed,'model':a.model,'elapsed_seconds':time.time()-start,'epochs':a.epochs,'targets':len(groups)*turns,'input_tokens_per_epoch':sum(len(r['input_ids']) for r in rows),'loss_tokens_per_epoch':sum(sum(v!=-100 for v in r['labels']) for r in rows),'optimizer_steps':completed_steps,'gradient_accumulation_steps':grad_accum,'completed':completed_steps==args.max_steps if args.max_steps>0 else trainer.state.epoch>=a.epochs},indent=2))

def norm(s):
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s.lower().translate(str.maketrans('','',string.punctuation))).split())
def score(p,refs):
    def f(ref):
        x=norm(p).split(); y=norm(ref).split(); c=sum((collections.Counter(x)&collections.Counter(y)).values())
        return (2*c/(len(x)+len(y))) if x and y else float(x==y)
    return max(float(norm(p)==norm(r)) for r in refs),max(f(r) for r in refs)

def evaluate(a):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    manifest=json.loads(Path(a.data+'/manifest.json').read_text()); gs=read(a.data+'/eval.jsonl')
    if a.adapter:
        run_path=Path(a.adapter)/'run.json'
        if not run_path.exists(): raise ValueError(f'Adapter has no completion metadata: {run_path}')
        run=json.loads(run_path.read_text())
        if run.get('epochs')!=a.expected_epochs or run.get('optimizer_steps')!=a.expected_steps or run.get('completed',True) is not True: raise ValueError(f'Adapter is not a complete {a.expected_epochs}-epoch/{a.expected_steps}-step run')
        if run.get('model')!=a.model: raise ValueError('Adapter model does not match evaluation model')
    tok=AutoTokenizer.from_pretrained(a.model); cuda=torch.cuda.is_available()
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32)
    if a.adapter: model=PeftModel.from_pretrained(model,a.adapter)
    model.to('cuda' if cuda else 'cpu').eval(); rows=[]
    def predict(ms):
        encoded=tok.apply_chat_template(ms,tokenize=True,add_generation_prompt=True,return_tensors='pt')
        ids=(encoded['input_ids'] if isinstance(encoded,dict) else encoded).to(model.device)
        with torch.no_grad(): out=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=64,do_sample=False,pad_token_id=tok.eos_token_id)
        return tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()
    for gi,g in enumerate(gs,1):
        if gi%10==0: print(f'Evaluated {gi}/{len(gs)} groups',flush=True)
        for i in range(len(g['qa'])):
            modes=['single','multi_gold']
            if manifest['dataset']=='squad': modes.append('multi_repeat_passage_gold')
            for mode in modes:
                ms=(single(g,i,manifest['dataset'])[:-1] if mode=='single' else
                    (coqa_multi_prompt(g,i) if manifest['dataset']=='coqa' else
                     messages(g,i+1,repeat_passage=(mode=='multi_repeat_passage_gold')))[:-1])
                p=predict(ms); em,f1=score(p,g['qa'][i]['a']); rows.append({'group':g['id'],'turn':i+1,'mode':mode,'prediction':p,'references':g['qa'][i]['a'],'em':em,'f1':f1})
        if manifest['dataset']=='coqa':
            for i,qa in enumerate(g['qa']):
                p=predict(single(g,i,manifest['dataset'])[:-1]); em,f1=score(p,qa['a']); rows.append({'group':g['id'],'turn':i+1,'mode':'single','prediction':p,'references':qa['a'],'em':em,'f1':f1})
                p=predict(coqa_multi_prompt(g,i)); em,f1=score(p,qa['a']); rows.append({'group':g['id'],'turn':i+1,'mode':'multi_gold','prediction':p,'references':qa['a'],'em':em,'f1':f1})
        if manifest['dataset']=='squad':
            # Rollout variants differ only in whether every turn re-supplies the passage.
            for mode,repeat in [('multi_repeat_passage_rollout',True),('multi_rollout',False)]:
                history=[{'role':'system','content':SYSTEM}]
                for i,qa in enumerate(g['qa']):
                    history.append({'role':'user','content':('Passage:\n'+g['context']+'\n\n' if repeat or i==0 else '')+'Question: '+qa['q']})
                    p=predict(history); em,f1=score(p,qa['a']); rows.append({'group':g['id'],'turn':i+1,'mode':mode,'prediction':p,'references':qa['a'],'em':em,'f1':f1}); history.append({'role':'assistant','content':p})
        else:
            # CoQA rollout keeps its original conversation structure (passage once).
            history=[{'role':'system','content':SYSTEM}]
            for i,qa in enumerate(g['qa']):
                history.append({'role':'user','content':('Passage:\n'+g['context']+'\n\n' if i==0 else '')+'Question: '+qa['q']})
                p=predict(history); em,f1=score(p,qa['a']); rows.append({'group':g['id'],'turn':i+1,'mode':'multi_rollout','prediction':p,'references':qa['a'],'em':em,'f1':f1}); history.append({'role':'assistant','content':p})
    write(a.output,rows)
    modes=sorted({r['mode'] for r in rows})
    metrics={m:{k:sum(r[k] for r in rows if r['mode']==m)/sum(r['mode']==m for r in rows) for k in ('em','f1')} for m in modes}
    print(metrics,flush=True)

def compare(a):
    import numpy as np
    x=read(a.left); y=read(a.right); dx={(r['group'],r['turn'],r['mode']):r for r in x}; dy={(r['group'],r['turn'],r['mode']):r for r in y}
    if dx.keys()!=dy.keys(): raise ValueError('Evaluation targets differ')
    rng=np.random.default_rng(a.seed); report={}
    for mode in sorted({k[2] for k in dx}):
        ids=sorted({k[0] for k in dx if k[2]==mode}); delta=np.array([np.mean([dx[k]['f1']-dy[k]['f1'] for k in dx if k[0]==g and k[2]==mode]) for g in ids])
        boot=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
        report[mode]={'left_minus_right_f1':float(delta.mean()),'cluster_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'n_passages':len(ids)}
    Path(a.output).write_text(json.dumps(report,indent=2)); print(report)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','train','eval','compare']); p.add_argument('--dataset',choices=['squad','coqa'],default='squad'); p.add_argument('--model',default='Qwen/Qwen2.5-1.5B-Instruct'); p.add_argument('--data',default='data/squad'); p.add_argument('--seed',type=int,default=42); p.add_argument('--turns',type=int,default=3); p.add_argument('--train-groups',type=int,default=1000); p.add_argument('--eval-groups',type=int,default=100); p.add_argument('--max-length',type=int,default=2048); p.add_argument('--format',choices=['single','multi'],default='multi'); p.add_argument('--repeat-passage',action='store_true',help='include the passage in every user turn for multi-format training'); p.add_argument('--match-multi-updates',action='store_true',help='use gradient accumulation to match multi-format optimizer update count for single-format training'); p.add_argument('--loss',choices=['assistant','full'],default='assistant'); p.add_argument('--epochs',type=int,default=2); p.add_argument('--output',default='runs/model'); p.add_argument('--adapter'); p.add_argument('--expected-epochs',type=int,default=1); p.add_argument('--expected-steps',type=int,default=1000); p.add_argument('--left'); p.add_argument('--right'); a=p.parse_args(); {'prepare':prepare,'train':train,'eval':evaluate,'compare':compare}[a.command](a)
