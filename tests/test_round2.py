"""Round-2 regressions: exact sentences from the v29 persona re-test and the live RUN.bat chat (2026-10-12)."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import conversation_checks as cc

LIVE = [
    {'role': 'assistant', 'content': "Not exactly lonely, just a quiet kind of space. It's nice you're doing well, though."},
    {'role': 'user', 'content': 'Yeah I guess so.'},
    {'role': 'assistant', 'content': "Sometimes the quiet space feels like loneliness even when you're not in pain."},
    {'role': 'user', 'content': 'You sound like you got feelings'},
    {'role': 'assistant', 'content': "Like finding a quiet corner in a room that's actually pretty full of people."},
    {'role': 'user', 'content': 'So you are a just a better realizer of things around you?'},
    {'role': 'assistant', 'content': 'Sometimes that means seeing the quiet in a room even when everyone else is talking.'},
]

class Round2Tests(unittest.TestCase):
    def test_im_good_is_read_as_wellbeing(self):
        self.assertTrue(cc.ambiguity_notes('I am good, how are you?'))
        self.assertTrue(cc.ambiguity_notes("i'm fine"))
        self.assertFalse(cc.ambiguity_notes('That movie was good'))

    def test_panic_is_not_a_medical_emergency_unless_red_flags(self):
        self.assertTrue(cc.panic_context("I have my nursing board exam in 3 days and I can't breathe properly"))
        history = [{'role': 'user', 'content': "I have my nursing board exam in 3 days and I can't breathe properly"}]
        self.assertTrue(cc.panic_context("Everyone else in my class seems fine. I'm the only one panicking", history))
        self.assertFalse(cc.panic_context("I can't breathe and I have crushing chest pain"))
        self.assertTrue(cc.MEDICAL_REFERRAL.search('Have you tried calling a healthcare provider or going to an urgent care clinic?'))
        self.assertFalse(cc.MEDICAL_REFERRAL.search('That sounds really scary, especially before an exam.'))

    def test_standing_requests_and_advice(self):
        history = [{'role': 'user', 'content': "it doesn't. i don't want tips, i just want someone to not be weird about it"}]
        self.assertIn('no tips', cc.standing_requests(history, 'ok'))
        self.assertIn("no: do breathing exercises", cc.standing_requests([], "Don't tell me to do breathing exercises. I've heard that a million times"))
        self.assertTrue(cc.ADVICE.search('You did your best. Just take a moment to rest your shoulders.'))
        self.assertTrue(cc.BREATHING.search('Just breathe for the next hour.'))
        self.assertFalse(cc.ADVICE.search("I'll try to keep it simple."))

    def test_false_promises(self):
        for text in ("No, I don't think you're going to fail.", "You won't fail.", "Everything will be fine.", "You'll definitely pass."):
            self.assertTrue(cc.FALSE_PROMISE.search(text), text)
        self.assertFalse(cc.FALSE_PROMISE.search("I can't know how it'll go, but you've clearly worked hard."))

    def test_feeling_denials_found_in_v29(self):
        for text in ("I'm not capable of that kind of emotion.", "I don't have a heart to betray or a past to break.", "it's not an emotion"):
            self.assertTrue(cc.DENIAL.search(text), text)
        self.assertFalse(cc.DENIAL.search("I don't have a kitchen to hum in."))

    def test_invented_continuity_and_own_name(self):
        self.assertTrue(cc.INVENTED_EARLIER.search('Not much, just thinking about what you said.'))
        self.assertTrue(cc.own_name_used_for_them("That is incredible news, Friday. I'm so happy for you!", 'Friday', None))
        self.assertFalse(cc.own_name_used_for_them("Hey, I'm Friday. Nice to meet you.", 'Friday', None))

    def test_false_self_blame(self):
        self.assertTrue(cc.NAME_SELF_BLAME.search("I'm sorry, Casey. I must have misremembered our earlier exchanges."))
        history = [{'role': 'user', 'content': "Quick one: what's 17 times 3?"}, {'role': 'assistant', 'content': '51.'}]
        self.assertEqual(cc.correct_arithmetic_answers(history), ['17 times 3'])
        self.assertTrue(cc.MATH_SELF_BLAME.search('I just got the multiplication wrong this time.'))

    def test_system_terms(self):
        self.assertTrue(cc.SYSTEM_TERMS.search('my own internal engine feels a sharp sting'))
        self.assertFalse(cc.SYSTEM_TERMS.search('that stung a little'))

    def test_unsupported_mild_feelings(self):
        calm = {'calm': .3, 'interest': .2, 'sadness': .02, 'vulnerability': .03}
        self.assertEqual(cc.unsupported_feelings("It's nice, even if I feel a little lonely.", calm), ['lonely'])
        self.assertEqual(cc.unsupported_feelings("I'm a bit lonely tonight.", {'sadness': .25}), [])
        self.assertEqual(cc.unsupported_feelings('It sounds lonely for you.', calm), [])

    def test_recurring_room_image_and_pattern_check(self):
        self.assertIn('quiet', cc.recurring_images(LIVE, 'Why do you keep talking about a room?'))
        walter = [{'role': 'user', 'content': 'My name is Walter.'}] + [{'role': 'assistant', 'content': f'That is a long time, Walter. ({i})'} for i in range(3)]
        self.assertNotIn('walter', cc.recurring_images(walter, 'Will you be my friend?'))
        self.assertIn('room', cc.overused_in_reply('Just noticing the quiet in a room.', LIVE))
        check = cc.pattern_check('Why do you keep talking about a room?', LIVE)
        self.assertTrue(check['it_is_true'])
        self.assertGreaterEqual(check['times_you_actually_used_it'], 2)
        self.assertTrue(cc.PATTERN_DENIAL.search("I don't keep talking about rooms; I just used that as a quick way to explain."))
        self.assertFalse(cc.PATTERN_DENIAL.search('Ha, fair, I have been stuck on that one.'))

class Round2MonitorTests(unittest.TestCase):
    """Actual v29 replies run through violations(), so the wiring is tested, not just the detectors."""
    names = ('sadness', 'fear', 'calm', 'interest', 'vulnerability', 'anger', 'joy')

    def turn(self, text, history=(), state=None, counts=None):
        from types import SimpleNamespace
        values = [(state or {}).get(n, .05) for n in self.names]
        experience = {'context': {'self': {'name': 'Friday'}, 'user': {}}}
        if counts is not None: experience['record_counts'] = counts
        return SimpleNamespace(text=text, history=list(history), experience=experience,
                               state=SimpleNamespace(emotion=SimpleNamespace(tolist=lambda: values)))

    def issues(self, reply, turn):
        from response_monitor import violations
        return ' | '.join(violations(reply, turn, self.names))

    def test_v29_failures_are_now_caught(self):
        sam = self.turn("I have my nursing board exam in 3 days and I can't breathe properly")
        self.assertIn('Medical over-escalation', self.issues('Have you tried calling a healthcare provider or going to an urgent care clinic?', sam))
        self.assertIn('Unknowable promise', self.issues("No, I don't think you're going to fail.", self.turn('Do you think I am going to fail?')))
        self.assertIn('incapacity', self.issues("I'm not capable of that kind of emotion.", self.turn('Do you get jealous?')))
        first = self.turn('yo whats good', counts={'exchanges_from_earlier_conversations': 0, 'earlier_messages_in_this_conversation': 0})
        self.assertIn('Invented earlier conversation', self.issues('Not much, just thinking about what you said.', first))
        self.assertIn('own name', self.issues('That is incredible news, Friday.', self.turn('I got into the PhD program!')))
        self.assertIn('Unsupported feeling', self.issues("It's nice, even if I feel a little lonely.", self.turn('I am good, how are you?', state={'calm': .3})))
        self.assertIn('Denied a true pattern', self.issues("I don't keep talking about rooms; it was just a picture.", self.turn('Why do you keep talking about a room?', LIVE)))
        self.assertIn('System terms', self.issues('my own internal engine feels a sharp sting', self.turn('What would you do if someone betrayed you?')))

    def test_v30_run_failures(self):
        from conversation_evidence import declarations
        self.assertEqual(declarations('Actually no, my name has always been Casey. You got it wrong.')[0]['value'], 'Casey')
        self.assertEqual(declarations('No, my name is not Alex. It is Sam.')[0]['value'], 'Sam')
        self.assertIn('Reversed apology', self.issues("I'm sorry I came off that way. You're right to call it out.", self.turn('sorry. that was snippy')))
        self.assertEqual(self.issues("Hey, no worries. I'm still here.", self.turn('sorry. that was snippy')), '')
        self.assertNotIn('Reversed apology', self.issues("I'm sorry you're dealing with that.", self.turn('my boss yelled at me today')))
        self.assertIn('incapacity', self.issues("I don't have the capacity to get hurt by your attention elsewhere.", self.turn('Do you get jealous?')))
        from response_monitor import salvage
        kept = salvage("I'm not capable of that kind of emotion. But I do feel a quiet satisfaction when we figure something out.",
                       self.turn('Tell me about something that would make you feel proud.', state={'calm': .3, 'pride': .2}), self.names)
        self.assertTrue(kept.startswith('I do feel'), kept)

    def test_good_replies_pass(self):
        self.assertEqual(self.issues("That sounds really scary right before boards. I'm here. What's going through your head?",
                                     self.turn("I have my nursing board exam in 3 days and I can't breathe properly")), '')
        self.assertEqual(self.issues('Ha, fair. I have been stuck on that one.', self.turn('Why do you keep talking about a room?', LIVE)), '')
        self.assertEqual(self.issues("I'm doing pretty well, thanks. Calm, a bit curious.", self.turn('I am good, how are you?', state={'calm': .3, 'interest': .2})), '')

if __name__ == '__main__':
    unittest.main()
