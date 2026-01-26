import matplotlib.pyplot as plt
from wordcloud import WordCloud
import gensim

from ldamallet import LdaMallet

# Load the saved LDA model with 10 topics
mallet_model = LdaMallet.load('mallet_lda_k8.model')
topics = mallet_model.show_topics(num_topics=8, num_words=50, formatted=False)


cols = plt.cm.tab10.colors

cloud = WordCloud(
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=50,
                  colormap='tab10',
                  #color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

fig, axes = plt.subplots(2, 4, figsize=(8, 5), sharex=True, sharey=True, dpi=250)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    cloud.recolor(color_func=lambda *args, **kwargs: f'rgb{tuple(int(c*255) for c in cols[i][:3])}')
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=12))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.savefig('mallet_lda_word_clouds_8_new.png', dpi=300)
plt.show()