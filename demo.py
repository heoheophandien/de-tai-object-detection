import torch
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageEnhance
from itertools import permutations, combinations
import numpy as np
import uuid
import random
import os
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s_heo.pt', force_reload=True) 

path_52cards = "52cards/"
prefix = "800px-Playing_card_"
suffix = ".svg.png"

suits = {"spade":0, "club":1, "diamond":2, "heart":3}
numbs = {None:0, "2":1, "3":2, "4":3, "5":4, "6":5, "7":6, "8":7,
         "9":8, "10":9, "J":10, "Q":11, "K":12, "A":13}

exp_14 = [1, 14, 196, 2744, 38416, 537824]

ranks = {
    "high_card": 0,
    "pair": 1,
    "two_pairs": 2,
    "three_of_a_kind": 3,
    "straight": 4,
    "flush": 5,
    "full_house": 6,
    "four_of_a_kind": 7,
    "straight_flush": 8,
}

A2345_combo = ("2", "3", "4", "5", "A")
all_of_straights = [("2", "3", "4", "5", "6")]
all_of_straights += [("3", "4", "5", "6", "7")]
all_of_straights += [("4", "5", "6", "7", "8")]
all_of_straights += [("5", "6", "7", "8", "9")]
all_of_straights += [("6", "7", "8", "9", "10")]
all_of_straights += [("7", "8", "9", "10", "J")]
all_of_straights += [("8", "9", "10", "J", "Q")]
all_of_straights += [("9", "10", "J", "Q", "K")]
all_of_straights += [("10", "J", "Q", "K", "A")]

all_of_straights_3ele = [("2", "3", "A")]
all_of_straights_3ele += [("2", "3", "4")]
all_of_straights_3ele += [("3", "4", "5")]
all_of_straights_3ele += [("4", "5", "6")]
all_of_straights_3ele += [("5", "6", "7")]
all_of_straights_3ele += [("6", "7", "8")]
all_of_straights_3ele += [("7", "8", "9")]
all_of_straights_3ele += [("8", "9", "10")]
all_of_straights_3ele += [("9", "10", "J")]
all_of_straights_3ele += [("10", "J", "Q")]
all_of_straights_3ele += [("J", "Q", "K")]
all_of_straights_3ele += [("Q", "K", "A")]


def card_split(suit_card):
    tmp = suit_card.split("_")
    return tmp[0], tmp[1]


def identify_combo(combo): # card_1 < card_2 < .. < card_5
    if len(combo) == 3:
        card_1, card_2 = None, None
        card_3, card_4, card_5 = combo
        s1, n1 = None, None
        s2, n2 = None, None
    else:
        card_1, card_2, card_3, card_4, card_5 = combo
        s1, n1 = card_split(card_1)
        s2, n2 = card_split(card_2)
    
    s3, n3 = card_split(card_3)
    s4, n4 = card_split(card_4)
    s5, n5 = card_split(card_5)
    
    
    # Straight flush ----------------------------|
    if s1 == s2 == s3 == s4 == s5:
        if combo in all_of_straights:
            return "straight_flush", n5, None, None, None, None
        if combo == A2345_combo:
            return "straight_flush", n4, None, None, None, None
        # Flush ---------------------------------|
        return "flush", n5, n4, n3, n2, n1
    
    # Four of a kind ----------------------------|
    if n1 == n4:
        return "four_of_a_kind", n4, None, None, None, None
    if n2 == n5:
        return "four_of_a_kind", n5, None, None, None, None
    
    # Full House --------------------------------|
    if n1 == n3 and n4 == n5:
        return "full_house", n3, None, None, None, None
    if n1 == n2 and n3 == n5:
        return "full_house", n5, None, None, None, None

    # Straight ----------------------------------|
    if combo == A2345_combo:
        return "straight", n4, None, None, None, None # it means that combo is 2-3-4-5-A
    if combo in all_of_straights:
        return "straight", n5, None, None, None, None
    
    # Three of a kind ---------------------------|
    if n1 == n3:
        return "three_of_a_kind", n3, None, None, None, None
    if n2 == n4:
        return "three_of_a_kind", n4, None, None, None, None
    if n3 == n5:
        return "three_of_a_kind", n5, None, None, None, None
    
    # Two pairs ---------------------------------|
    if n1 == n2 and n1 != None:
        if n3 == n4:
            return "two_pairs", n4, n2, n5, None, None
        if n4 == n5:
            return "two_pairs", n5, n2, n3, None, None
    else:
        if n2 == n3 and n4 == n5:
            return "two_pairs", n5, n3, n1, None, None
    
    # Pair --------------------------------------|
    if n1 == n2 and n1 != None:
        return "pair", n2, n5, n4, n3, None
    if n2 == n3:
        return "pair", n3, n5, n4, n1, None
    if n3 == n4:
        return "pair", n4, n5, n2, n1, None
    if n4 == n5:
        return "pair", n5, n3, n2, n1, None
    
    # High card ---------------------------------|
    return "high_card", n5, n4, n3, n2, n1


def numbs_comparison(n1, n2):
    if n1 == n2 == None:
        return 0
    if n1 == None:
        return -1
    if n2 == None:
        return 1
    
    if numbs[n1] < numbs[n2]:
        return -1
    if numbs[n1] > numbs[n2]:
        return 1
    return 0
    

def comboes_comparison(combo_1, combo_2):
    identify_1, r11, r12, r13, r14, r15 = identify_combo(combo_1)
    identify_2, r21, r22, r23, r24, r25 = identify_combo(combo_2)
    i_in_1 = ranks[identify_1]
    i_in_2 = ranks[identify_2]
    if i_in_1 > i_in_2:
        return 1
    if i_in_1 < i_in_2:
        return -1
    
    if identify_1 == "straight_flush":
        return numbs_comparison(r11, r21)
    
    if identify_1 == "four_of_a_kind":
        return numbs_comparison(r11, r21)
    
    if identify_1 == "full_house":
        return numbs_comparison(r11, r21)
    
    if identify_1 == "flush":
        comp_1 = numbs_comparison(r11, r21)
        if comp_1 != 0:
            return comp_1
        comp_2 = numbs_comparison(r12, r22)
        if comp_2 != 0:
            return comp_2
        comp_3 = numbs_comparison(r13, r23)
        if comp_3 != 0:
            return comp_3
        comp_4 = numbs_comparison(r14, r24)
        if comp_4 != 0:
            return comp_4
        return numbs_comparison(r15, r25)
    
    if identify_1 == "straight":
        return numbs_comparison(r11, r21)
        
    if identify_1 == "three_of_a_kind":
        return numbs_comparison(r11, r21)
    
    if identify_1 == "two_pairs":
        comp_1 = numbs_comparison(r11, r21)
        if comp_1 != 0:
            return comp_1
        comp_2 = numbs_comparison(r12, r22)
        if comp_2 != 0:
            return comp_2
        return numbs_comparison(r13, r23)
    
    if identify_1 == "pair":
        comp_1 = numbs_comparison(r11, r21)
        if comp_1 != 0:
            return comp_1
        comp_2 = numbs_comparison(r12, r22)
        if comp_2 != 0:
            return comp_2
        comp_3 = numbs_comparison(r13, r23)
        if comp_3 != 0:
            return comp_3
        return numbs_comparison(r14, r24)
    
    # identify_1 == "high card"
    comp_1 = numbs_comparison(r11, r21)
    if comp_1 != 0:
        return comp_1
    comp_2 = numbs_comparison(r12, r22)
    if comp_2 != 0:
        return comp_2
    comp_3 = numbs_comparison(r13, r23)
    if comp_3 != 0:
        return comp_3
    comp_4 = numbs_comparison(r14, r24)
    if comp_4 != 0:
        return comp_4
    return numbs_comparison(r15, r25)


def hands_comparison(hands_1, hands_2):
    combo_1_1, combo_1_2, combo_1_3 = hands_1
    combo_2_1, combo_2_2, combo_2_3 = hands_2
    value = comboes_comparison(combo_1_1, combo_2_1)
    value += comboes_comparison(combo_1_2, combo_2_2)
    value += comboes_comparison(combo_1_3, combo_2_3)
    return value


def scores_computation(combo):
    identify, r1, r2, r3, r4, r5 = identify_combo(combo)
    score = ranks[identify] + numbs[r1]/exp_14[1] + numbs[r2]/exp_14[2] \
        + numbs[r3]/exp_14[3] + numbs[r4]/exp_14[4] + numbs[r5]/exp_14[5]
    return score


def permutations_553(hands):
    
    # Check Chinese Poker perfect win
    hands_suit = dict()
    hands_num = dict()
    for card in hands:
        suit, num = card_split(card)
        if suit not in hands_suit.keys():
            hands_suit[suit] = 1
        else:
            hands_suit[suit] += 1
        if num not in hands_num.keys():
            hands_num[num] = 1
        else:
            hands_num[num] += 1
            
    perfect_win = len(hands_num) == 13 # from 2 to A

    if perfect_win:
        combo_1 = (hands[0], hands[1], hands[2], hands[3], hands[4])
        combo_2 = (hands[5], hands[6], hands[7], hands[8], hands[9])
        combo_3 = (hands[10], hands[11], hands[12])
        hand = [combo_1, combo_2, combo_3]
        return [hand], [float('inf')]
    
    perm_553 = []
    scores_553 = []
    comboes_1 = list(combinations(hands, 5))
    for c1, c2, c3, c4, c5 in comboes_1:
        combo_1 = (c1, c2, c3, c4, c5)
        tmp_hands = hands.copy()
        tmp_hands.remove(c1)
        tmp_hands.remove(c2)
        tmp_hands.remove(c3)
        tmp_hands.remove(c4)
        tmp_hands.remove(c5)
        
        comboes_2 = list(combinations(tmp_hands, 5))
        for c6, c7, c8, c9, c10 in comboes_2:
            combo_2 = (c6, c7, c8, c9, c10)

            combo_3 = tmp_hands.copy()
            combo_3.remove(c6)
            combo_3.remove(c7)
            combo_3.remove(c8)
            combo_3.remove(c9)
            combo_3.remove(c10)
            combo_3 = tuple(combo_3)
            
            score_1 = scores_computation(combo_1)
            score_2 = scores_computation(combo_2)
            score_3 = scores_computation(combo_3)
            
            ident_1 = int(score_1)
            ident_2 = int(score_2)
            ident_3 = int(score_3)

            if score_1 >= score_2 and score_2 >= score_3:
                vip_hands = [combo_1, combo_2, combo_3]
                
                perm_553.append(vip_hands)
                if ident_1 == ident_2 == 5:  # two of flushes
                    card_1, card_2, card_3 = combo_3
                    s1, n1 = card_split(card_1)
                    s2, n2 = card_split(card_2)
                    s3, n3 = card_split(card_3)
                    if s1 == s2 == s3:
                        score = float('inf')
                elif ident_1 == ident_2 == 4: # two of straights
                    if combo_3 in all_of_straights_3ele:
                        score = float('inf')
                elif ident_1 == 6 and ident_2 == 6: # two of full houses
                    s1, n1 = card_split(card_1)
                    s2, n2 = card_split(card_2)
                    s3, n3 = card_split(card_3)
                    if n1 == n3: # ofcourse equal n2
                        score = float('inf')
                else:
                    score = score_1 + score_2 + score_3
                scores_553.append(score)

    return perm_553, scores_553


def sort_n2(a):
    
    def cards_comparison(card_1, card_2):
        s1, c1 = card_split(card_1)
        s2, c2 = card_split(card_2)
        if numbs[c1] > numbs[c2]:
            return 1
        elif numbs[c1] < numbs[c2]:
            return -1
        else:
            if suits[s1] > suits[s2]:
                return 1
            elif suits[s1] < suits[s2]:
                return -1
        return 0
    
    
    for i in range(0, len(a)-1):
        for j in range(i+1, len(a)):
            if cards_comparison(a[i], a[j]) > 0:
                a[i], a[j] = a[j], a[i]


def partition(A, left_index, right_index, B):
    pivot = A[left_index]
    i = left_index + 1
    for j in range(left_index + 1, right_index):
        if A[j] > pivot:
            A[j], A[i] = A[i], A[j]
            B[j], B[i] = B[i], B[j]
            i += 1
    A[left_index], A[i - 1] = A[i - 1], A[left_index]
    B[left_index], B[i - 1] = B[i - 1], B[left_index]
    return i - 1


def quick_sort_random(A, left, right, B):
    if left < right:
        pivot = random.randint(left, right - 1)
        A[pivot], A[left] = A[left], A[pivot] # switches the pivot with the left most bound
        B[pivot], B[left] = B[left], B[pivot]
        pivot_index = partition(A, left, right, B)
        quick_sort_random(
            A, left, pivot_index, B
        )  # recursive quicksort to the left of the pivot point
        quick_sort_random(
            A, pivot_index + 1, right, B
        )  # recursive quicksort to the right of the pivot point


dragon_path = path_52cards + "dragon.png"
dragon_img = mpimg.imread(dragon_path)
test_path = "test/"
results_path = "results/"
os.makedirs(test_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)
khang_path = path_52cards + "KhangBao.jpg"
khang_img = mpimg.imread(khang_path)

def xapxam(img):
    try:
        uuid_generator = str(uuid.uuid4())
        image_path = test_path + uuid_generator + ".jpg"
        result_path = results_path + uuid_generator + ".jpg"

        width, height = img.size
        if width == 0 or height == 0:
            return khang_img
        if height > 640:
            width = width * 640 // height
            height = 640
            img = img.resize((width, height))

    #     enhancer = ImageEnhance.Brightness(img)
    #     factor = 1.0
    #     img = enhancer.enhance(factor)

        img_save = img.save(image_path)

        predictions = model(image_path).pandas().xyxy[0]

        classes = []
        for card_pred in predictions.name:
            if card_pred not in classes:
                classes.append(card_pred)
            if len(classes) == 13:
                break
        print(classes)
        if len(classes) < 13:
            return khang_img

        hands = []
        for cl in classes:
            card = cl[:-1]
            if cl[-1] == "S":
                card = "spade_" + card
            elif cl[-1] == "C":
                card = "club_" + card
            elif cl[-1] == "D":
                card = "diamond_" + card
            elif cl[-1] == "H":
                card = "heart_" + card
            hands.append(card)

        sort_n2(hands)

    
        hands_553, scores_553 = permutations_553(hands)
        quick_sort_random(scores_553, 0, len(scores_553), hands_553)

        hand, sc = hands_553[0], scores_553[0]
        fig, ax = plt.subplots()
        fig.set_figwidth(18)
        fig.set_figheight(12)
        if sc == float('inf'):
            ax.imshow(dragon_img)

        plt.axis("off")
        columns, rows = 5, 3
        for i in range(2):
            for j in range(5):
                card = hand[i][j]
                card_path = path_52cards + prefix + card + suffix
                img = mpimg.imread(card_path)
                fig.add_subplot(rows, columns, i*5 + j + 1)
                plt.axis("off")
                plt.imshow(img)
        i = 2
        blank_path = path_52cards + "blank.png"
        img = mpimg.imread(blank_path)
        fig.add_subplot(rows, columns, i*5 + 0 + 1)
        plt.axis("off")
        plt.imshow(img)
        fig.add_subplot(rows, columns, i*5 + 4 + 1)
        plt.axis("off")
        plt.imshow(img)
        for j in range(1, 3+1):
            card = hand[i][j-1]
            card_path = path_52cards + prefix + card + suffix
            img = mpimg.imread(card_path)
            fig.add_subplot(rows, columns, i*5 + j + 1)
            plt.axis("off")
            plt.imshow(img)

        fig.savefig(result_path)
        im = Image.open(result_path)
    
    except Exception as e:
        im = mpimg.imread(khang_path)
        print(e)
    return im


demo = gr.Interface(
    xapxam, 
    inputs=[gr.Image(type="pil", label="Input Image"),], 
    outputs="image",
)

demo.launch(share=True)