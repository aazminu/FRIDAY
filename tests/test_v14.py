"""Regression checks use synthetic vectors and temporary checkpoints only."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v14_model',ROOT/'src/model.py')
v14=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=v14
spec.loader.exec_module(v14)

class V14Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)
        words=list(v14.SPECIAL)+list(dict.fromkeys(v14.ANCHORS+('hello','world','help','answer')))
        self.vocab=v14.Vocabulary(words)
        self.agent=v14.V14Agent(self.vocab,torch.randn(len(words),24),v14.Config(d=24,heads=4,max_tokens=32))
        self.needs=self.agent.deficits([.5]*7)
        self.ids=torch.tensor(self.vocab.encode('hello help'))

    def test_core(self):
        v14.self_test()

    def test_recurrent_training_and_frequency_counts(self):
        examples=[{'input':'hello help','action':'ANSWER','response':'answer world','episode':'a'},
                  {'input':'hello world','action':'COMFORT','response':'help hello','episode':'a'}]
        before=self.agent.gut_cell.weight_ih.detach().clone()
        v14.train_examples(self.agent,examples,epochs=2,tbptt=2)
        self.assertFalse(torch.equal(before,self.agent.gut_cell.weight_ih))
        self.assertEqual(self.agent.counts[self.vocab.ids['hello']].item(),3)

    def test_invalid_measurements_and_nonterminal_bootstrap(self):
        for q in ([.5]*6,[float('nan')]*7,[1.1]*7):
            with self.assertRaises(ValueError): self.agent.deficits(q)
        turn=self.agent(self.ids,self.agent.initial_state(),self.needs)
        with self.assertRaises(ValueError):
            v14.learn_outcome(self.agent,turn,0,'answer',{'levels':[.5]*7,'task_reward':1.,'terminal':False})
        state,_,stats=v14.learn_outcome(self.agent,turn,0,'answer',
            {'levels':[.5]*7,'task_reward':1.,'terminal':False,'next_input':'hello world'})
        self.assertTrue(torch.isfinite(state.emotion).all())
        self.assertGreater(stats['reward'],0)

    def test_scar_writes_require_trained_direction_and_measured_outcome(self):
        turn=self.agent(self.ids,self.agent.initial_state(),self.needs)
        turn.state.emotion[v14.EMOTIONS.index('pain')]=1
        turn.state.emotion[v14.EMOTIONS.index('fear')]=1
        self.agent.commit_outcome(turn,self.needs)
        self.assertEqual(self.agent.scars.abs().sum().item(),0)
        label={'input':'hello help','action':'ANSWER','response':'answer',
               'levels':[.5]*7,'next_levels':[.55]*7,'scar_direction':[1.]*24}
        v14.train_examples(self.agent,[label],1)
        self.assertTrue(self.agent.scar_ready.item())
        self.agent.commit_outcome(turn,self.needs)
        self.assertGreater(self.agent.scars.abs().sum().item(),0)
        self.assertTrue((self.agent.scars.norm(dim=-1)<=self.agent.cfg.scar_max).all())

    def test_intent_changes_language_and_candidate_constraints(self):
        turn=self.agent(self.ids,self.agent.initial_state(),self.needs)
        seq=torch.tensor([[2,self.vocab.ids['hello']]])
        self.assertFalse(torch.allclose(self.agent.language_logits(turn,0,seq),self.agent.language_logits(turn,1,seq)))
        output=self.agent.candidates(turn,0,validator=lambda text:False)
        self.assertIn('No candidate passed',output)

    def test_legacy_checkpoint_protection(self):
        with self.assertRaises(ValueError):
            v14.save_checkpoint(self.agent,self.agent.initial_state(),ROOT/'state/legacy/vantage_brain.pth')

    def test_dataset_loader_does_not_execute_python(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'training.py'
            path.write_text("TRAINING_DATA = [{'input':'hello','action':'ANSWER','response':'world'}]\nraise RuntimeError('do not execute')",encoding='utf-8')
            self.assertEqual(len(v14.load_examples(path)),1)
            path.write_text("TRAINING_DATA = list()",encoding='utf-8')
            with self.assertRaises(ValueError): v14.load_examples(path)

    def test_partial_emotion_labels_and_validation_without_updates(self):
        example={'input':'hello help','action':'ANSWER','response':'answer world','emotion':{'joy':.8}}
        base={key:value for key,value in example.items() if key!='emotion'}
        base_loss,turn=v14.supervised_loss(self.agent,base,self.agent.initial_state())
        labeled_loss,_=v14.supervised_loss(self.agent,example,self.agent.initial_state())
        expected=.2*(turn.proposal[v14.EMOTIONS.index('joy')]-.8).square()
        self.assertTrue(torch.allclose(labeled_loss-base_loss,expected,atol=1e-6))
        before={key:value.clone() for key,value in self.agent.state_dict().items()}
        result=v14.evaluate_examples(self.agent,[example])
        self.assertEqual(result['examples'],1)
        self.assertTrue(all(torch.equal(value,self.agent.state_dict()[key]) for key,value in before.items()))
        example['emotion']={'energy':.9}
        with self.assertRaises(ValueError): v14.supervised_loss(self.agent,example,self.agent.initial_state())

    def test_project_datasets_are_disjoint(self):
        training=v14.load_examples(ROOT/'data/training/training_data.py')
        validation=v14.load_examples(ROOT/'data/training/validation_data.py')
        meaning=v14.load_examples(ROOT/'data/training/meaning_validation.py')
        self.assertFalse({e['input'] for e in training}&{e['input'] for e in validation})
        self.assertEqual(set(e['action'] for e in training),set(v14.ACTIONS))
        self.assertTrue(all('next_levels' not in e and 'levels' not in e for e in training))
        self.assertFalse({e['input'] for e in training}&{e['input'] for e in meaning})
        self.assertEqual(len(training),180)
        self.assertEqual(len(validation),50)
        self.assertTrue(all(sum(e['action']==a for e in training)==36 for a in v14.ACTIONS))
        # Duplicate prompts are permitted only when the history/episode differs.
        signatures=[(e['input'],str(e.get('history',[])),e.get('episode')) for e in training]
        self.assertEqual(len(signatures),len(set(signatures)))

    def test_appraisal_contrast_ties_do_not_count_as_understanding(self):
        base={'input':'hello help','action':'ANSWER','response':'answer world',
              'contrast_group':'same-context','contrast_channel':'fear'}
        examples=[dict(base,contrast_rank='high',emotion={'fear':.7}),
                  dict(base,contrast_rank='low',emotion={'fear':.1})]
        result=v14.evaluate_examples(self.agent,examples)
        self.assertEqual(result['contrast_pairs'],1)
        self.assertEqual(result['contrast_order_accuracy'],0)
        self.assertAlmostEqual(result['contrast_mean_margin'],0)
        self.assertEqual(result['appraisal_labels'],2)
        with self.assertRaises(ValueError): v14.evaluate_examples(self.agent,examples[:1])

    def test_pretrained_adapter_with_local_tiny_fixture(self):
        # A random tiny fixture tests plumbing, not pretrained language quality.
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config,GPT2LMHeadModel,PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as directory:
            tokenizer=Tokenizer(WordLevel(self.vocab.ids,unk_token='<unk>'))
            tokenizer.pre_tokenizer=Whitespace()
            fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='<unk>',
                bos_token='<bos>',eos_token='<eos>',pad_token='<pad>')
            fast.chat_template="{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\n' }}{% endfor %}assistant:"
            fast.save_pretrained(directory)
            lm=GPT2LMHeadModel(GPT2Config(vocab_size=len(self.vocab.words),n_embd=24,
                n_layer=1,n_head=4,n_positions=64,bos_token_id=2,eos_token_id=3,
                resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.))
            lm.save_pretrained(directory)
            cfg=v14.Config(d=24,heads=4,max_tokens=32,pretrained_decoder=directory,contextual_appraisal=True)
            agent=v14.V14Agent(self.vocab,self.agent.semantic.weight.detach().clone(),cfg)
            agent.eval()
            turn=agent(self.ids,agent.initial_state(),self.needs)
            seq=torch.tensor([agent.response_ids('hello world')])
            changed=seq.clone(); changed[0,-1]=self.vocab.ids['help']
            logits=agent.language_logits(turn,0,seq)
            self.assertTrue(torch.allclose(logits[:,:-1],agent.language_logits(turn,0,changed)[:,:-1],atol=1e-6))
            loss=torch.nn.functional.cross_entropy(logits[0,:-1],seq[0,1:])
            loss.backward()
            self.assertGreater(agent.decoder.bridge.weight.grad.norm().item(),0)
            self.assertGreater(agent.gut_cell.weight_ih.grad.norm().item(),0)
            self.assertGreater(agent.decoder.context_projection.weight.grad.norm().item(),0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in agent.decoder.lm.parameters()))
            agent.train()
            self.assertFalse(agent.decoder.lm.training)
            before={k:v.clone() for k,v in agent.decoder.lm.state_dict().items()}
            bridge=agent.decoder.bridge.weight.detach().clone()
            optimizer=torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad),lr=.001)
            optimizer.step()
            self.assertFalse(torch.equal(bridge,agent.decoder.bridge.weight))
            self.assertTrue(all(torch.equal(v,agent.decoder.lm.state_dict()[k]) for k,v in before.items()))
            agent.eval()
            turn=agent(self.ids,agent.initial_state(),self.needs,text='hello world')
            other=agent(self.ids,agent.initial_state(),self.needs,text='help world')
            self.assertFalse(torch.allclose(turn.proposal,other.proposal,atol=1e-8,rtol=0))
            self.assertFalse(torch.allclose(agent.language_logits(turn,0,seq),agent.language_logits(other,0,seq)))
            other=agent(self.ids,agent.initial_state(),self.needs,text='hello world',
                        history=[{'role':'user','content':'help'},{'role':'assistant','content':'answer'}])
            self.assertFalse(torch.allclose(agent.language_logits(turn,0,seq),agent.language_logits(other,0,seq)))
            self.assertFalse(torch.allclose(turn.proposal,other.proposal,atol=1e-8,rtol=0))
            reordered=agent(self.ids,agent.initial_state(),self.needs,text='world hello')
            self.assertFalse(torch.allclose(turn.proposal,reordered.proposal,atol=1e-8,rtol=0))
            self.assertFalse(torch.allclose(agent.language_logits(turn,0,seq),agent.language_logits(turn,0,seq,use_bridge=False)))
            expected=[agent.decoder.tokenizer.eos_token_id]
            for _ in range(5):
                token=agent.language_logits(turn,0,torch.tensor([expected]))[0,-1].argmax().item()
                if token==agent.decoder.tokenizer.eos_token_id: break
                expected.append(token)
            cached=agent.generate(turn,0,max_new=5)
            self.assertEqual(cached,agent.decoder.tokenizer.decode(expected[1:],skip_special_tokens=True).strip())
            long=agent(self.ids,agent.initial_state(),self.needs,text='hello '*200)
            prefix=seq[:,:1]
            self.assertTrue(torch.allclose(agent.language_logits(long,0,prefix)[:,0],
                                           agent.language_logits(long,0,seq)[:,0],atol=1e-6))
            agent.chat_history=[{'role':'user','content':'hello'}]
            path=Path(directory)/'checkpoint.pth'
            v14.save_checkpoint(agent,agent.initial_state(),path)
            payload=torch.load(path,weights_only=True)
            self.assertFalse(any(k.startswith('decoder.lm.') for k in payload['model']))
            self.assertTrue(payload['frozen_base'])
            restored,restored_state=v14.load_checkpoint(path)
            self.assertEqual(agent.chat_history,restored.chat_history)
            self.assertTrue(torch.allclose(agent.language_logits(turn,0,seq),restored.language_logits(turn,0,seq)))
            # Windows must allow replacing the same checkpoint after memory-mapped loading.
            v14.save_checkpoint(restored,restored_state,path)
            # Changed source weights must not silently load an adapter trained on another base.
            payload['frozen_base']['config.json']='wrong'
            torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,'source weights changed'): v14.load_checkpoint(path)

if __name__=='__main__': unittest.main()
