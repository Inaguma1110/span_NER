# def nest_triangle_cut(span1,span2,span3,span4):
#     for index in range(0, len(span4)):
#         try:
#             if span4[index] == 1:
#                 span3[index] = span3[index+1] = span3[index+2] = span3[index+3] = 0
#                 span2[index] = span2[index+1] = span2[index+2] = span2[index+3] = 0
#                 span1[index] = span1[index+1] = span1[index+2] = span1[index+3] = 0
#                 continue
#             else:
#                 if span3[index] == 1:
#                     span2[index] = span2[index+1] = span2[index+2] = 0
#                     span1[index] = span1[index+1] = span1[index+2] = 0
#                     continue
#                 else:
#                     if span2[index] == 1:
#                         span1[index] = span1[index+1] = 0
#                         continue
#         except IndexError:
#             pass
#     return span1, span2, span3, span4


import pdb
import torch 
def nest_square_cut_for_eval(span1,span2,span3,span4):
    for index in range(0, len(span4)):
        try:
            if span4[index] == 1:
                span3[index] = span3[index+1] = span3[index+2] = span3[index+3] = 0
                span2[index] = span2[index+1] = span2[index+2] = span2[index+3] = 0
                span1[index] = span1[index+1] = span1[index+2] = span1[index+3] = 0
                continue
            else:
                if span3[index] == 1:
                    span2[index] = span2[index+1] = span2[index+2] = 0
                    span1[index] = span1[index+1] = span1[index+2] = 0
                    continue
                else:
                    if span2[index] == 1:
                        span1[index] = span1[index+1] = 0
                        continue
        except IndexError:
            pass
    return span1, span2, span3, span4


def nest_square_cut(pred1,pred2,pred3,pred4):
    for m in range(0, len(pred4[0])):
        for index in range(0, len(pred4[0][0])):
            try:
                if pred4[0][m][index] == 1:
                    pred3[0][m][index] = pred3[0][m][index+1] = pred3[0][m][index+2] = pred3[0][m][index+3] = 0
                    pred2[0][m][index] = pred2[0][m][index+1] = pred2[0][m][index+2] = pred2[0][m][index+3] = 0
                    pred1[0][m][index] = pred1[0][m][index+1] = pred1[0][m][index+2] = pred1[0][m][index+3] = 0
                    continue
                else:
                    if pred3[0][m][index] == 1:
                        pred2[0][m][index] = pred2[0][m][index+1] = pred2[0][m][index+2] = 0
                        pred1[0][m][index] = pred1[0][m][index+1] = pred1[0][m][index+2] = 0
                        continue
                    else:
                        if pred2[0][m][index] == 1:
                            pred1[0][m][index] = pred1[0][m][index+1] = 0
                            continue
            except IndexError:
                pass
    return pred1, pred2, pred3, pred4


def nest_cut(spans, span_size): #spans = (batch, span_size, seq_len)
    for m in range(len(spans)):
        for s in reversed(range(len(spans[m]))):
            for index in range(len(spans[m][s])):
                if spans[m][s][index] == 1:
                    for i in range(span_size):
                        if s-i-1>=0:
                            spans[m][s-i-1][index:index+s+1] = 0
    return spans