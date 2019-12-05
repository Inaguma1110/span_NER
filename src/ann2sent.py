import pdb
import os
import pdb
inputpath = "/home/inaguma.19406/span_NER_RE/"
outputpath = "/home/inaguma.19406/span_NER_RE/files/target_data/"

filelst = os.listdir(inputpath)

doclist=[]

for tfile in filelst:
    docpair = ()
    if (tfile[-4:]) == '.txt':
        for afile in filelst:
            if (afile[-4:]) == '.ann' and afile[:-4] == tfile[:-4]:
                docpair = (tfile, afile)
                doclist.append(docpair)

# -----------------------------make entity dictionary and relation dictionary------------------------------------------
#pdb.set_trace()
for x in range(len(doclist)):
    begtotal = 0
    endtotal = 0
    num = 0
    reldict = {}
    entdict = {}

    txtfile = open(inputpath + doclist[x][0], "r")
    annfile = open(inputpath + doclist[x][1], "r")
    for aline in annfile:
        ann = []
        ann = aline.split("\t")
        ID = ann[0]
        if(ID[0] != "#"):
            if(ID[0] == "T"):

                tag_beg_end = ann[1]
                tag,beg,end = tag_beg_end.split(" ")
                surface = ann[2]
                entdict[ID] = (int(beg),int(end)-1,tag,surface)

            if(ID[0] == "R"):
                tag_Arg1_Arg2 = ann[1]
                tag,Arg1,Arg2 = tag_Arg1_Arg2.split(" ")

                reldict[ID] = (Arg1,Arg2,tag)

#------------------------------------------make sentence map ----------------------------------------------------------
    sentmap = []
    #pdb.set_trace()
    for i, tline in enumerate(txtfile):
        num = len(tline)
        endtotal += num

        entmap = []
        linemap = []

        for e_k, e_v in entdict.items():
            begword = e_v[0]
            endword = e_v[1]
            if begtotal <= begword  < endtotal and begtotal < endword <= endtotal:
                entmap.append(e_k)
        linemap = [i,entmap,(begtotal,endtotal)]
        begtotal = endtotal
        sentmap.append(linemap)

    #pdb.set_trace()

# --------------------------------- make list of an ID of the sentence that relation exist in -------------------------
#    rel_exist_text_id = []
#    rel_exist_dict = {}
#    for r_k, r_v in reldict.items():
#        Argtag1 = r_v[0][5:]
#        Argtag2 = r_v[1][5:]
#        #print(Argtag1,Argtag2)
#        for j, tagmap in enumerate(sentmap):
#            if Argtag1 in tagmap[1] and Argtag2 in tagmap[1]:
#                rel_exist_text_id.append(j)
#                rel_exist_dict[r_k] = r_v
#    rel_exist_text_id.sort()
#    rel_exist_text_id = list(set(rel_exist_text_id))
#------------------------------- make ann file -------------------------------------------------------------------------
    txtfile = open(inputpath + doclist[x][0], "r")
    annfile = open(inputpath + doclist[x][1], "r")
    begtotal_out = 0
    endtotal_out = 0
    for k, tlineout in enumerate(txtfile):
        sent_len = len(tlineout)
        endtotal_out += sent_len

        #pdb.set_trace()
        f = open(outputpath + doclist[x][0][:-4] + "_" + "{:0=4}".format(x) + "_" + "{:0=3}".format(k) + ".txt", "w")
        f.write(tlineout)
        f.close()
        #pdb.set_trace()
        
        for key, value in entdict.items():
            sta_ent = int(value[0])
            end_ent = int(value[1])
            ent_tag = value[2]
            ent_surface = value[3]

            if begtotal_out <= sta_ent < endtotal_out and begtotal_out < end_ent <= endtotal_out:
                sta_ent = sta_ent - begtotal_out
                end_ent = end_ent - begtotal_out
                g = open(outputpath + doclist[x][1][:-4] + "_" + "{:0=4}".format(x) + "_" + "{:0=3}".format(k) + ".ann", "a")
                g.write(key + "\t" + ent_tag + " " + str(sta_ent) + " " + str(end_ent + 1) + "\t" + ent_surface)
                for rel_k, rel_v in reldict.items():
                    Argtag1out = rel_v[0][5:]
                    Argtag2out = rel_v[1][5:]
                    if Argtag1out == key:
                        g.write(rel_k + "\t" + rel_v[2] + " " + rel_v[0] + " " + rel_v[1] + "\n")
                g.close()
        begtotal_out = endtotal_out
