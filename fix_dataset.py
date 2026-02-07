import pandas as pd
import random
import uuid
from datetime import datetime, timedelta

# Templates for generating more natural, context-rich sentences
positive_templates = [
    "I really {pos_adj} this product, it works {pos_adv}.",
    "The service was {pos_adj} and the staff was {pos_adj}.",
    "I am so {pos_adj} with my purchase!",
    "Definitely {pos_adj}, I would recommend it to everyone.",
    "It was not {neg_adj}, it was actually {pos_adj}.",
    "Despite the delay, the result was {pos_adj}.",
    "I have never seen such a {pos_adj} performance.",
    "Just {pos_adj}! I love it.",
    "This is the {pos_adj} thing I have ever bought.",
    "Super {pos_adj} experience, will come back again.",
    "I became {pos_adj} wicket taker in this match",
    "I scored {pos_adj} runs in this match",
    "I was {pos_adj} in this match",
    "I have my first {pos_adj} in this match.",
    "I have my first {pos_adj} in Australia.",
]

negative_templates = [
    "I am very {neg_adj} with this service.",
    "The quality is {neg_adj} and not {pos_adj} at all.",
    "I did not like it, it was {neg_adj}.",
    "This is {neg_adj}, do not buy it.",
    "I thought it would be {pos_adj}, but it was {neg_adj}.",
    "Not {pos_adj} at all, very {neg_adj}.",
    "Worst experience ever, completely {neg_adj}.",
    "I am {neg_adj} regarding the support I received.",
    "It stopped working after one day, simply {neg_adj}.",
    "I regret buying this, it is {neg_adj}."
]

neutral_templates = [
    "The product is {neu_adj}, nothing special.",
    "It is {neu_adj}, does the job but that's it.",
    "Just a {neu_adj} day, feeling {neu_adj}.",
    "I have no strong opinion, it is {neu_adj}.",
    "It is neither {pos_adj} nor {neg_adj}, just {neu_adj}.",
    "Standard quality, pretty {neu_adj}.",
    "It arrived today, looks {neu_adj}.",
    "Not sure how I feel, seems {neu_adj}.",
    "An {neu_adj} experience overall.",
    "It is okay, just {neu_adj}."
]

# Vocabulary
pos_adj = [
    'good', 'century', 'great', 'excellent', 'highest', 'amazing', 'wonderful','outstanding' , 'fantastic', 'superb', 'brilliant', 'perfect', 'awesome',
    'lovely', 'delighted', 'pleased', 'impressive', 'terrific', 'fabulous', 'incredible', 'stunning', 'beautiful',
    'satisfied', 'glad', 'nice', 'excited', 'joyful', 'happy', 'best', 'superior', 'top-notch', 'first-class'
]
pos_adv = ['well', 'perfectly', 'beautifully', 'splendidly', 'wonderfully', 'amazingly', 'incredibly', 'absolutely', 'truly', 'really']

neg_adj = [
    'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor', 'useless', 'pathetic', 'dreadful', 'nasty',
    'disgusting', 'annoying', 'frustrating', 'painful', 'garbage', 'trash', 'fail', 'boring', 'miserable', 'unhappy',
    'worst', 'inferior', 'subpar', 'lousy', 'abysmal', 'atrocious', 'unpleasant', 'regrettable', 'shameful', 'broken'
]

neu_adj = [
    'okay', 'average', 'fine', 'normal', 'standard', 'regular', 'typical', 'fair', 'moderate', 'decent',
    'plain', 'basic', 'routine', 'expected', 'general', 'ordinary', 'passable', 'unchanged', 'steady', 'balanced',
    'middle', 'flat', 'mediocre', 'common', 'simple', 'usual', 'acceptable', 'ok', 'so-so', 'undistinguished'
]

# Slang and Verbs
SLANG_POSITIVE = ["sick", "wicked", "crazy", "insane", "ill", "badass", "fire", "lit", "goated"]
pos_verbs = ["work", "load", "connect", "function", "respond", "perform", "launch", "operate"]
neg_verbs = ["crash", "lag", "freeze", "disconnect", "fail", "break", "bug out", "glitch"]

# Add new templates
positive_templates.extend([
    "The new features are {slang_pos}.",
    "Honestly this update is {slang_pos}.",
    "This is absolutely {slang_pos}!",
    "Just {slang_pos}, nothing else to say."
])

negative_templates.extend([
    "I would love it if the app actually {pos_verb}.",
    "It would be great if it didn't {neg_verb}.",
    "I wish it was {pos_adj}, but it is not.",
    "Great concept, but {neg_adj} execution.",
    "Supposed to be {pos_adj}, turned out {neg_adj}.",
    "Why does it always {neg_verb} when I need it?"
])

def generate_post(sentiment):
    if sentiment == 'Positive':
        template = random.choice(positive_templates)
    elif sentiment == 'Negative':
        template = random.choice(negative_templates)
    else:
        template = random.choice(neutral_templates)
        
    # Fill blanks
    post = template.format(
        pos_adj=random.choice(pos_adj),
        pos_adv=random.choice(pos_adv),
        neg_adj=random.choice(neg_adj),
        neu_adj=random.choice(neu_adj),
        slang_pos=random.choice(SLANG_POSITIVE),
        pos_verb=random.choice(pos_verbs),
        neg_verb=random.choice(neg_verbs)
    )
    
    return post

data = []
for _ in range(3000): # Increased size slightly
    sentiment = random.choice(['Positive', 'Negative', 'Neutral'])
    post_content = generate_post(sentiment)
    
    label = sentiment
    if random.random() < 0.015:
        label = random.choice([s for s in ['Positive', 'Negative', 'Neutral'] if s != sentiment])
    
    data.append({
        'Post ID': str(uuid.uuid4()),
        'Post Content': post_content,
        'Sentiment Label': label,
        'Number of Likes': random.randint(0, 1000),
        'Number of Shares': random.randint(0, 500),
        'Number of Comments': random.randint(0, 200),
        'User Follower Count': random.randint(100, 10000),
        'Post Date and Time': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d %H:%M:%S"),
        'Post Type': random.choice(['text', 'image', 'video']),
        'Language': 'en'
    })

df = pd.DataFrame(data)
df.to_csv('synthetic_social_media_data.csv', index=False)
print("New context-rich synthetic dataset generated.")
