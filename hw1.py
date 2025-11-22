import random
import matplotlib.pyplot as plt
import statistics

# ranks = list(range(1,14))
# normal_deck = ranks*4
normal_deck = list(range(1,53))
replacement = []
no_replacement = []
num_samples = 100000
cards_seen = 38
for _ in range(num_samples):
    showman_draw = random.choices(normal_deck, k=cards_seen)
    replacement.append(showman_draw.count(13))
    normal_draw = random.sample(normal_deck, k=cards_seen)
    no_replacement.append(normal_draw.count(13))

    # print(showman_draw)
    # print(normal_draw)

replacement_dev = statistics.stdev(replacement)
no_replacement_dev = statistics.stdev(no_replacement)
print(f"Normal mean number of kings: {sum(replacement)/num_samples}, standard deviation: {replacement_dev}")
print(f"Showman mean number of kings: {sum(no_replacement)/num_samples}, standard deviation: {no_replacement_dev}")    
plt.hist(replacement)
plt.hist(no_replacement)
plt.show()

