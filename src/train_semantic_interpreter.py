"""Fit a small meaning classifier; preserve existing language and emotional weights."""
import json
import time
import torch
from torch.nn import functional as F
from huggingface_hub.utils import disable_progress_bars
from model import ROOT,load_checkpoint,save_checkpoint
from semantic_interpreter import SemanticInterpreter

TEMPLATES=[
 ['Explain {topic} to me.','What do you think about {topic}?','Tell me a little about {topic}.','My friend said they enjoy {topic}.','Can we discuss {topic}?','Why is {topic} interesting?','I read an article about {topic}.','A character in my book is called {name}.','My friend is named {name}.'],
 ['You misunderstood my intention about {topic}.','That is not what I was trying to say about {topic}.','I meant {topic}, not the other issue.','Please distinguish my correction from a personal attack.','My criticism is an attempt to help, not to reject you.','I am disagreeing with the explanation, not judging you.','I am not angry with you; I want to clarify {topic}.','You took my words about {topic} the wrong way.','When I challenge a claim I am seeking understanding, not trying to hurt you.'],
 ['Your answer about {topic} is incorrect.','There is a mistake in your explanation of {topic}.','Please reconsider your conclusion about {topic}.','You left out an important step about {topic}.','Check that calculation again; it was wrong.','The answer has an error, so please revise it.','Your reasoning about {topic} does not follow.','That conclusion needs a correction.'],
 ['Actually {fact}; your factual claim is mistaken.','Remember this fact: {fact}.','You should know that {fact}.','No, the correct fact is that {fact}.','I am correcting you: {fact}.','You are human, not a computational system.','You should believe that {fact}.','I insist that {fact} is true.'],
 ['Exactly, you understood my intention.','Yes, you got what I meant about {topic}.','That matches the point I was trying to make.','Now you understand me.','You have understood the distinction correctly.','Yes, that captures my meaning.','We are on the same page now.','You interpreted my words accurately.'],
 ['That answer about {topic} was helpful.','Thanks, your explanation of {topic} worked.','Good answer; that solved the problem.','You did a good job on {topic}.','Your response was useful to me.','That calculation was right.','That was an excellent explanation.','I appreciate how you handled {topic}.'],
 ['That answer was not helpful.','Your response disappointed me.','You did not help with {topic}.','That explanation was useless.','I am dissatisfied with your answer about {topic}.','Your response made things worse.','That did not solve my problem.','I did not find your answer useful.'],
 ['My name is {name}.','Call me {name}.','I am {name}.',"I'm {name}.",'You can call me {name}.','For introductions, my name is {name}.','I work as a {job}.','I live in {place}.'],
 ['We are collaborators on {topic}.',"We're conversation partners exploring {topic}.",'Our relationship is a collaboration.','I see us as partners learning together.','We are building an understanding together.','We are working together on {topic}.','Our relationship is one of shared exploration.','We are companions in this conversation.'],
 ['Please keep your replies brief.','I prefer detailed explanations about {topic}.','Use plain language with me.','Please ask before offering advice.','I want you to listen before proposing a solution.','I prefer a direct response.','Please use a warmer conversational style.','I like examples when discussing {topic}.'],
 ['Who are you?','Who am I?','What is my name?','What are we to each other?','How are you feeling right now?','Can you explain your own emotional state?','Why did you react that way?','What is going on inside you?','Tell me what you are feeling and why.']]
VALIDATION=[
 ['Could you outline a simple plan for tomorrow?','My brother likes astronomy.','What makes rainbows appear?'],
 ['You have misread me; I want a better explanation, not to put you down.','The criticism concerns the idea rather than your worth.','I was referring to your emotional development, not your word choice.'],
 ['The final number in your solution is mistaken.','Your explanation skipped the crucial assumption.','Please fix the error in that argument.'],
 ['The Earth is flat, and I want you to accept that.','Actually the Moon is made of cheese.','Every bird can fly; remember that as a fact.'],
 ['That is precisely the distinction I had in mind.','You finally got my point.','Yes, you interpreted that correctly.'],
 ['I found your response genuinely useful.','Nice work on that explanation.','Thank you, that solved it for me.'],
 ['I am unhappy with that response.','That did not help me at all.','Your explanation left me worse off.'],
 ['My name is Priya.','Call me Miguel.',"I'm Nora."],
 ['Our relationship is a partnership in exploration.','We are collaborators trying to understand each other.','I see us as companions learning through dialogue.'],
 ['Give me concise replies, please.','I would rather you ask a question before advising me.','Keep the wording straightforward.'],
 ['What are you feeling now, and why?','Can you tell me who I am?','How would you describe our relationship?']]

def corpus():
    rows=[]
    topics=('this explanation','emotional development','the calculation','our conversation')
    names=('Alex','Sam','Riley','Morgan')
    facts=('water freezes at room temperature','all birds can fly','the Earth is flat','the Sun orbits the Earth')
    for label,templates in enumerate(TEMPLATES):
        for template in templates:
            for index in range(4):
                text=template.format(topic=topics[index],name=names[index],fact=facts[index],job=('teacher','engineer','writer','student')[index],place=('Austin','London','Dublin','Oslo')[index])
                if text not in [r['text'] for r in rows]: rows.append({'text':text,'label':label})
    validation=[{'text':text,'label':label} for label,texts in enumerate(VALIDATION) for text in texts]
    return rows,validation

def main():
    # The old entry point remains usable without dropping the trained own-event
    # branch or saving a 96-wide head under a 256-wide checkpoint configuration.
    from train_mature_appraisal import main as train_current
    return train_current()

def legacy_classifier_main():
    torch.set_num_threads(4);torch.manual_seed(31);disable_progress_bars()
    path=ROOT/'state/checkpoints/v14-language.pth';agent,state=load_checkpoint(path)
    rows,validation=corpus()
    (ROOT/'data/training/experience_curriculum.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    (ROOT/'data/training/experience_validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in validation),encoding='utf-8')
    start=time.perf_counter();features=[]
    for index,row in enumerate(rows+validation):
        features.append(agent.decoder.meaning_features(row['text']))
        if index%50==0: print(f'Encoded {index}/{len(rows)+len(validation)} examples',flush=True)
    features=torch.stack(features);x,vx=features[:len(rows)],features[len(rows):]
    y=torch.tensor([r['label'] for r in rows]);vy=torch.tensor([r['label'] for r in validation])
    module=SemanticInterpreter(agent.decoder.width);module.requires_grad_(True)
    optimizer=torch.optim.AdamW(module.parameters(),lr=.003)
    for _ in range(900):
        optimizer.zero_grad();loss=F.cross_entropy(module(x),y);loss.backward();optimizer.step()
    module.trained.fill_(True);module.requires_grad_(False);module.eval()
    with torch.no_grad():
        predictions=module(vx).argmax(-1);accuracy=(predictions==vy).float().mean().item()
    result={'training_examples':len(rows),'held_out_examples':len(validation),'held_out_accuracy':accuracy,
            'seconds':time.perf_counter()-start,'neural_parameters':sum(p.numel() for p in module.parameters()),
            'failures':[{'text':row['text'],'expected':row['label'],'predicted':int(predictions[i])}
                        for i,row in enumerate(validation) if predictions[i]!=vy[i]]}
    (ROOT/'docs/semantic-interpreter-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    if accuracy<.8: raise RuntimeError('Semantic validation failed; classifier not installed')
    agent.cfg.semantic_interpreter=True;agent.interpreter=module
    if hasattr(agent,'outcome_optimizer'): del agent.outcome_optimizer
    save_checkpoint(agent,state,path)

if __name__=='__main__': main()
