import unittest
from bench import messages,single,encode,score
class FakeTokenizer:
    def apply_chat_template(self,ms,**kwargs):
        return [v for m in ms for v in [ {'system':1,'user':2,'assistant':3}[m['role']],len(m['content']),9]]
class Tests(unittest.TestCase):
    def test_pairs_and_masks(self):
        g={'context':'Ada was born in London.','qa':[{'q':'Who?','a':['Ada']},{'q':'Where?','a':['London']},{'q':'Birthplace?','a':['London']}]}
        ms=messages(g); self.assertEqual(sum(m['role']=='assistant' for m in ms),3)
        x=encode(FakeTokenizer(),ms,'assistant'); self.assertEqual(sum(l!=-100 for l in x['labels']),9)
        for i,m in enumerate(ms): self.assertEqual(x['labels'][i*3:(i+1)*3],x['input_ids'][i*3:(i+1)*3] if m['role']=='assistant' else [-100]*3)
        self.assertNotIn('Previous conversation',single(g,1,'squad')[1]['content'])
        self.assertIn('Q: Who?',single(g,1,'coqa')[1]['content'])
        self.assertEqual(encode(FakeTokenizer(),ms,'full')['labels'],x['input_ids'])
    def test_metrics(self):
        self.assertEqual(score('The London.', ['London']), (1.0,1.0))
        self.assertEqual(score('Paris',['London']), (0.0,0.0))
if __name__=='__main__': unittest.main()
