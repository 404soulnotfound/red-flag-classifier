"""
Synthetic dataset generator for Red Flag Text Classifier.
Generates labeled text examples across red flag categories and green flags.
"""

import sys
import pandas as pd
import random

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─── RED FLAG EXAMPLES ────────────────────────────────────────────────────────

GASLIGHTING = [
    "You're imagining things, that never happened.",
    "You're way too sensitive, I was just joking.",
    "I never said that. You must be misremembering.",
    "You're being dramatic again, calm down.",
    "No one else has a problem with me, it's just you.",
    "You're making things up to start a fight.",
    "That's not what I meant and you know it.",
    "You always twist my words.",
    "You're crazy if you think I would do that.",
    "I can't believe you'd accuse me of something like that.",
    "You need to stop letting your emotions control you.",
    "Everyone agrees with me, you're the odd one out.",
    "You're remembering it wrong, as usual.",
    "Stop overreacting, it was nothing.",
    "Why do you always make everything into a big deal?",
    "I'm not raising my voice. You're imagining it.",
    "You're too emotional to see things clearly.",
    "This is exactly why I don't tell you things.",
    "You always do this — blow things out of proportion.",
    "It didn't happen the way you think it did.",
]

LOVE_BOMBING = [
    "I've never felt this way about anyone in my life, ever.",
    "You're literally the only person who understands me.",
    "I want to marry you and I've known you for 3 days.",
    "I've been waiting my whole life for someone like you.",
    "You're perfect, don't ever change.",
    "I can't imagine life without you already.",
    "Let's move in together, I know it's fast but it feels right.",
    "You're my soulmate, I knew it the moment I saw you.",
    "I'd do absolutely anything for you, anything.",
    "No one has ever made me feel this complete.",
    "I love you more than words can say and we just met.",
    "You're the only good thing in my life right now.",
    "I think about you every single second of every day.",
    "I've never connected with anyone this fast, it must be fate.",
    "You make me want to be a better person, stay with me forever.",
    "I already told my whole family about you.",
    "Let's make it official right now, I don't want to wait.",
    "You're unlike anyone I've ever met, you're extraordinary.",
    "I want to spend every moment with you.",
    "My friends say I'm moving too fast but they don't understand us.",
]

CONTROLLING = [
    "Who is that guy in your photo? Delete it now.",
    "You don't need to go out with your friends tonight.",
    "I need to know where you are at all times.",
    "Why didn't you answer my call? I called 10 times.",
    "You're not wearing that, it's too revealing.",
    "You spend too much time with your family.",
    "I don't like you talking to him. Stop.",
    "Send me your location right now.",
    "You should ask me before making plans.",
    "I don't think you should take that job opportunity.",
    "Who were you texting just now? Let me see your phone.",
    "You need to come home right now.",
    "Cancel your plans, I want to see you tonight.",
    "I don't want you being friends with her anymore.",
    "You need my permission before spending money like that.",
    "Why are you dressing up? Who are you trying to impress?",
    "Tell your friends you can't hang out, you're with me.",
    "I read your messages. You shouldn't be talking to him.",
    "You have to check in with me every hour.",
    "Delete his number right now if you care about me.",
]

GUILT_TRIPPING = [
    "Fine, don't worry about me. I'll be fine alone.",
    "I guess I'm just not important enough to you.",
    "After everything I've done for you, this is how you treat me?",
    "I sacrifice everything and you can't do this one thing.",
    "If you really cared you wouldn't have to think twice.",
    "Don't mind me, I'll just sit here by myself.",
    "I never ask for anything and the one time I do...",
    "I guess my feelings don't matter to you.",
    "You'd do it if you actually loved me.",
    "I always put you first. You never do the same.",
    "You're choosing them over me? Wow.",
    "I went through so much trouble and you don't even appreciate it.",
    "I had a terrible day and you're not even here for me.",
    "Every time I need you, you disappear.",
    "I thought you were different but you're just like everyone else.",
    "Fine, it's okay. I'll just deal with it alone like always.",
    "I do everything for this relationship and you do nothing.",
    "No one ever puts me first, not even you.",
    "I'm sorry for being such a burden to you.",
    "You make me feel like I'm invisible.",
]

MANIPULATION = [
    "If you loved me, you wouldn't go out tonight.",
    "I'll hurt myself if you break up with me.",
    "You owe me after everything I've put up with.",
    "I know things about you that nobody else does. Remember that.",
    "If you tell anyone, I'll make sure people know your secrets.",
    "You're nothing without me and you know it.",
    "Either you do this or we're done.",
    "No one else will ever love you like I do.",
    "If you leave me, I'll tell everyone what you did.",
    "You need me more than you think you do.",
    "I could have anyone I want, you should feel lucky.",
    "You'd be lost without me in your life.",
    "I made you who you are. Don't forget that.",
    "Do this or I'm leaving, your choice.",
    "You'll regret losing me. I promise you that.",
    "Nobody else would put up with you the way I do.",
    "I'll change, but only if you stay. That's the deal.",
    "Fine, then we're done. See how you like being alone.",
    "You think you can do better? Good luck.",
    "You need to prove to me that you love me.",
]

DISRESPECT = [
    "You're so stupid, how do you not understand this?",
    "Nobody asked for your opinion.",
    "Shut up, you don't know what you're talking about.",
    "God, you're so annoying sometimes.",
    "You're embarrassing yourself right now.",
    "I can't believe how dumb that was.",
    "You're a joke, you know that?",
    "Don't talk to me like you know things.",
    "Your ideas are always so ridiculous.",
    "You're the most exhausting person I've ever met.",
    "You act like a child when you don't get your way.",
    "You're pathetic.",
    "I can't take you seriously when you act like this.",
    "Everyone thinks you're difficult to deal with, not just me.",
    "Why are you even here if you're going to be like this?",
    "You have no idea what you're doing.",
    "Stop acting like you matter in this situation.",
    "You're such a burden.",
    "I regret getting involved with you.",
    "You're the problem, not me.",
]

# ─── GREEN FLAG EXAMPLES ──────────────────────────────────────────────────────

GREEN_FLAG = [
    "I understand you're upset. Can we talk about what happened?",
    "Take all the time you need with your friends tonight, have fun!",
    "I respect your boundaries and I won't push you on this.",
    "Thank you for telling me how you feel, it really helps me understand.",
    "I'm sorry, I was wrong. That was my fault.",
    "Your career goals matter a lot — I support you fully.",
    "I trust you. I don't need to know every detail.",
    "Hey, I noticed you seemed off. Are you okay?",
    "It's completely fine that you need space sometimes.",
    "Let's talk this through calmly and figure it out together.",
    "Your feelings are valid, even if I see it differently.",
    "I love spending time with you but I know you need time for yourself too.",
    "I'm proud of you for going after that opportunity.",
    "No, go hang out with your family. We'll catch up later.",
    "If you ever feel uncomfortable, just say the word and we'll leave.",
    "I value your opinion even when we disagree.",
    "Let's find a compromise that works for both of us.",
    "I appreciate everything you do, even the small things.",
    "I'm here for you no matter what you're going through.",
    "You don't have to explain yourself to me. I trust you.",
    "I want you to maintain your friendships — they're important.",
    "Tell me what you need right now. I'll do my best.",
    "It's okay to say no. I'll never pressure you.",
    "I'm glad you have people in your life who care about you.",
    "Let's check in with each other — how are you really feeling?",
    "I love how independent and driven you are.",
    "We can disagree and still respect each other.",
    "You were right and I should have listened. I'm sorry.",
    "I never want you to feel like you have to hide things from me.",
    "Your happiness matters to me, even when I'm not the reason for it.",
    "Take your time, there's no rush at all.",
    "I completely understand why you feel that way.",
    "You don't owe me anything, do what feels right for you.",
    "I'm glad you told me honestly instead of holding it in.",
    "Go ahead and wear whatever makes you feel confident!",
    "I'm cheering for you, you've worked so hard for this.",
    "If you need a quiet night to yourself, I totally get it.",
    "Let's figure out what works for both of us, no pressure.",
    "I hear you, and I'll do better next time.",
    "You have every right to be upset about that.",
    "I trust your judgment completely.",
    "Thank you for being patient with me while I learn.",
    "I'm really happy you have such supportive friends.",
    "Text me whenever you're free, no need to drop everything.",
    "I'd never want you to miss out on something important to you.",
    "You're allowed to change your mind anytime.",
    "I appreciate how honest and open you are with me.",
    "Let's take a breather and talk when we're both ready.",
    "Your boundaries are completely reasonable.",
    "I'm sorry I interrupted you, please finish what you were saying.",
    "Whatever you decide, I've got your back.",
    "You don't have to apologize for needing space.",
    "I value your time and I know you're busy today.",
    "Tell me if I ever cross a line so I can fix it.",
    "I'm excited for you to spend time with your family.",
    "We're a team, we'll get through this together.",
    "Thank you for trusting me with how you feel.",
]


def build_dataset():
    data = []

    categories = {
        "gaslighting": GASLIGHTING,
        "love_bombing": LOVE_BOMBING,
        "controlling": CONTROLLING,
        "guilt_tripping": GUILT_TRIPPING,
        "manipulation": MANIPULATION,
        "disrespect": DISRESPECT,
    }

    for category, texts in categories.items():
        for text in texts:
            data.append({
                "text": text,
                "category": category,
                "label": 1,          # 1 = Red Flag
                "label_name": "red_flag"
            })

    for text in GREEN_FLAG:
        data.append({
            "text": text,
            "category": "green_flag",
            "label": 0,              # 0 = Green Flag
            "label_name": "green_flag"
        })

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv("data/dataset.csv", index=False)
    print(f"✅ Dataset generated: {len(df)} examples")
    print(df["label_name"].value_counts())
    return df


if __name__ == "__main__":
    build_dataset()
