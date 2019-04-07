#this version allows members of the same designated clade occupying different communities to have different matched strains
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import itertools
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import re
import random

dat = pd.read_csv("geigerDevt.csv", header=0) # working CSV = datMLformatted_geiger2.csv
taxa = sorted(list(set(dat['NameOnPhylo'])))
wols = list(set(dat['wspClade']))
wols.remove('noWol')
comms = sorted(list(set(dat['fig.species'])))
X = np.array(dat[taxa])
    
###### SET SOME PARAMETERS!!!!! ######
increment = 100 #how many divisions btwn upper & lower to split "species"
upper = 4 #upper val - if upper == increment then spp delim range will cover upto 100% pw dists
lower = 0 #lower val - if lower == 0 then begin from 0% pw dists
purge = 6 #upper % for examining purging
z = len(str(purge)) #fao CSV column name house-keeping (i.e. zfill length)
pge_incr = 2 #purge increment - default = 1
prefix = 'pandaTEST' + '_Rel_' #add a prefix to file names - do NOT use 'x'
min_nSpp = 10 #min no. species from Kmeans
max_nSpp = 35 #max no. species from Kmeans

def main():
    '''
    Main pgm control function
    '''
    if 'x' in prefix:
        print('RUN STOPPED: Do NOT use an "x" in "prefix" variable!')
        return
    f = str(lower)+'-'+str(upper)+'x'+str(increment)+'_prg'+str(purge)
    print('Running "wolPredictor - Kmeans RELAXED" - params:', prefix, f)
    job = prefix + f + '_jobProgress.txt'
    assigned, taxonDesignations = pd.DataFrame(taxa, columns = ['taxa']), pd.DataFrame(taxa, columns = ['taxa'])
    assigned = assigned.join(dat['wspClade'])
    start = timer()
    for thresh in range(lower, upper):
        if thresh % (increment/10) == 0: print(job, 'Matrix iteration: ' + str(thresh) + ' - time: ' + str(timer() - start)[:-8])
        mat = np.array(calcMat((float(thresh)/increment), X))
        res = calcK(mat) #get optimal k-means cluster
        df = addPredict(res, assigned, thresh) #spDelim is delimited clusters in actual host comms
        tupl_df = list(df.itertuples(index = False)) #make df immutable
        taxonDesignations = taxonDesignations.join(pd.DataFrame(list(res.labels_), columns = ["thresh" + str(thresh).zfill(z) + '_nSpp' + str(len(set(res.labels_)))]))
        assigned = assigned.join(df[list(df)[1]])
        assigned.columns.values[-1] = "thresh" + str(thresh).zfill(z) + "_noPurge" 
        for thresh2 in range(0, purge + 1, pge_incr): assigned = wolPurger(assigned, pd.DataFrame(tupl_df), thresh2, thresh)

    assigned = pd.concat([assigned.iloc[:,:2], assigned.reindex(sorted(assigned.columns[2:]), axis=1)], axis = 1) #order columns
    assigned.to_csv(prefix + 'wolPreds_threshClades_incr' + f + '.csv', index = False)
    taxonDesignations.to_csv(prefix + 'taxonDesignations_' + f + '.csv', index = False)
    assigned = matchStrains(assigned, taxonDesignations, start, job)
    assigned.to_csv(prefix + 'correctedWolPreds_threshClades_incr' + f + '.csv', index = False)
    makePDF(f)
    print("Time:", str(timer() - start)[:-8])

def addPredict(res, assigned, thresh):
    '''
    predict Wolbachia strain accoring to rules based on clusters on indvs within a community
    indvs will be given different strains if there are >=2 different taxon designations in a community 
    '''
    wolClades, tmpTaxa = [], []
    labsDik = dict(zip(taxa, res.labels_))
    for comm in comms: #go thru fig host comms
        indvs = list(dat['NameOnPhylo'][dat['fig.species'] == comm]) #what indvs in that host community?
        [tmpTaxa.append(indv) for indv in indvs]
        clusters, grps = [], []
        [clusters.append(labsDik.get(indv)) for indv in indvs] #what barcode grps in that community?
        setClusts = list(set(clusters))
        if len(setClusts) > 1:
            [grps.append(comm[:3]+'_w'+str(setClusts.index(clusters[indvs.index(indv)]) + 1)) for indv in indvs]
        else:
            grps = ['noWol'] * len(indvs)
        [wolClades.append(grp) for grp in grps]
    col = 'thresh' + repr(thresh/increment)[2:]
    df1, df2 = pd.DataFrame(tmpTaxa, columns = ['tmpTaxa']), pd.DataFrame(wolClades, columns = [col])
    df = df1.join(df2)
    df = df.sort_values('tmpTaxa').reset_index(drop=True) #"reset_index" prevents a major fuck up

    return df #prev returned assigned

def wolPurger(assigned, dfm, thresh2, thresh):
    '''
    Decide whether divergence between species clusters warrants Wolbachia purging
        -at low thresh2 vals - most strains are purged (as most distances between clades are greater than thresh2)
        -at hi thresh2 vals - most are retained
    '''
    strainComms = list(set([x.split('_')[0] for x in dfm[list(dfm)[1]]])) #all communities with strains
    if 'noWol' in strainComms: strainComms.remove('noWol')
    to_purge, indv_strains, pairs = [], [], []
    for sc in strainComms: #eg ['arf', 'mic', 'tri']
        tmpStrains = []
        [tmpStrains.append(ss) for ss in dfm[list(dfm)[1]] if sc in ss ] #matching assigned strains
        lst = list(itertools.combinations(set(tmpStrains), 2)) #all unique strains in this community
        for l in lst: #get indv clade pairs where WOL is NOT required
            grp1, grp2 = list(dfm['tmpTaxa'][dfm[list(dfm)[1]] == l[0]]), list(dfm['tmpTaxa'][dfm[list(dfm)[1]] == l[1]])
            minInter = [dat.loc[dat.NameOnPhylo == g1, g2].tolist()[0] for g1 in grp1 for g2 in grp2]
            if min(minInter) >= thresh2/increment: indv_strains.append(l[0]), indv_strains.append(l[1]), pairs.append(sorted([l[0], l[1]])) #pairs indicates where wol is not required
 
    setStrains, sortPairs, strainComms = sorted(list(set(indv_strains))), [sorted(i) for i in pairs], list(set([x.split('_')[0] for x in indv_strains])) #strainComms = all communities with strains

    for sc in strainComms: #'tri', 'mic'
        matches = [string for string in setStrains if re.match(re.compile(sc + '.'), string)] #count no. clades/strains in this community
        for strain in setStrains: #['mic_w1', 'mic_w2', 'tri_w1', 'tri_w2', 'tri_w3']
            if sc in strain:
                cnt1 = sum([1 for pair in sortPairs if strain in pair]) #How many times does the clade/strain feature?
                if len(matches) - cnt1 < 2: to_purge.append(strain)

    for p in list(set(to_purge)): dfm[list(dfm)[1]].replace(p, 'noWol', inplace=True)

    dfm.rename(columns={list(dfm)[1]: 'thresh' + str(thresh).zfill(z) + '_purge_' + str(thresh2).zfill(z)}, inplace=True)
    assigned = assigned.join(dfm[list(dfm)[1]])

    return assigned

def matchStrains(assigned, taxonDesignations, start, job):
    '''
    match the predictions with the empirically (yet arbritarily) named strains as much as possible
    ISSUE: the 2nd block may not find a solution (e.g. thresh16_noPurge):
        -because it does not remove previously attempted suboptimal solutions (attempted if optimal soln doesn't solve)
        -i.e. 'replace_with' is retained and attempted again at the next iteration if the solution doesn't work
        -currently fudged to give up at 20 attempts (basically a heuristic search)
    '''
    for column in list(assigned)[2:]:
        if list(assigned)[2:].index(column) % (increment/10) == 0: print(job, 'Strain matching iteration: ' + str(column) + ' - time: ' + str(timer() - start)[:-8])
        tax_deg = [string for string in list(taxonDesignations) if re.match(re.compile(column.split('_')[0] + '.'), string)] #find spp delim at this thresh
        df = pd.concat([assigned.iloc[:,:2], taxonDesignations[tax_deg], assigned[column]], axis = 1) #df matching spp delim w pred & empirical strains
        tab = df.groupby([list(df)[1], list(df)[2], list(df)[3]]).size().reset_index(name = 'counts') #count combinations in df
        sel_preds, sel_wols, sel_cnts, output = [], [], [], True
        for wol in wols:
            sub_tab = tab.loc[tab['wspClade'] == wol].loc[tab[column] != 'noWol']
            if len(sub_tab) == 0: continue
            strain = sub_tab.loc[sub_tab['counts'] == max(sub_tab.loc[sub_tab['wspClade'] == wol]['counts']), column]
            strain = strain.iloc[random.randint(0, len(strain) - 1)] #if most common strains are tied randomly choose one
            sel_preds.append(strain), sel_wols.append(wol), sel_cnts.append(sub_tab.loc[sub_tab[column] == strain, 'counts'].item())
        dik, dik2 = dict(zip(sel_wols, sel_preds)), dict(zip(sel_wols, sel_cnts))

        cntr = 0
        while len(set(sel_wols)) != len(set(sel_preds)):
            if cntr == 20:
                print('NO SOLUTION FOUND!!:', column)
                output = False
                break
            probs = []
            [probs.append(pred) for pred in set(sel_preds) if sel_preds.count(pred) > 1]
            for prob in probs:
                tmp_wols, cnts = [], []
                [[tmp_wols.append(d), cnts.append(dik2.get(d))] for d in dik if dik.get(d) == prob] #dbl list comp - append 2 vars
                dik3 = dict(zip(tmp_wols, cnts)) #dik of duplicated strains
                prob_clade = list(sorted(dik3.items(), key=lambda x: x[1]))[:-1][-1][0]
                prob_tab = tab.loc[tab[column] != prob][tab['wspClade'] == prob_clade][tab[column] != 'noWol']
                if len(prob_tab) == 0: dik.pop(prob_clade, None) #if no options left remove strain
                else:
                    replace_with = prob_tab.loc[prob_tab['counts'] == max(prob_tab['counts'].values), column]
                    replace_with = replace_with.iloc[random.randint(0, len(replace_with) - 1)]
                    dik.update({prob_clade: replace_with})
                sel_preds, sel_wols =  [], [] 
                for d2 in dik: sel_wols.append(d2), sel_preds.append(dik.get(d2))
            cntr += 1

        if output:
            for sel_wol in sel_wols:
                sub_assigned = assigned.loc[assigned['wspClade'] == sel_wol]
                for taxon in sub_assigned[sub_assigned[column].isin([dik.get(sel_wol)])]['taxa']:
                    assigned.loc[assigned.taxa == taxon, column] = sel_wol

    return assigned

def calcMat(thresh, X):
    '''
    create a binary matrix according to < or > threshold of cophenetic distances
    '''
    return (X > thresh).astype(int) #or return mat.astype(int)

#calculate clusters of individuals according to kmeans from calcMat 
def calcK(mat):
    '''
    delimit species by Kmeans and select "best" solution
    '''
    Ks = range(min_nSpp, max_nSpp)
    km = [KMeans(n_clusters=i, init='k-means++') for i in Ks]
    scores = [k.fit(mat).score(mat) for k in km]
    diffs, props =  [], []
    [diffs.append(i-scores[scores.index(i)+1]) for i in scores[:-1]]
    [props.append(float(i)/(diffs[diffs.index(i)+1]+0.00000000001)) for i in diffs[:-1]] #prevent zero divisors

    return km[props.index(max(props)) + 1] #is this always the inflection point???

def makePDF(f):
    '''
    write PDF of result
    '''
    f = prefix + 'correctedWolPreds_threshClades_incr' + f
    print("Figure title:", f)
    props=[]
    incr = 100/float(f.split('x')[1].split('_')[0])
    prg = f.split('prg')[1]
    dat = pd.read_csv(f + ".csv", header=0)
    runs = list(dat)[2:]
    maximum=0
    for run in runs:
        cnt, tot = 0, 0
        for cell in dat[run]:
            if cell == dat.loc[cnt][1]:
                tot += 1
            cnt += 1
        props.append((tot/cnt)*100)
        if tot > maximum:
            maximum = tot
            bestThr = run[6:]
    print("maximum", maximum/cnt*100,"%")
    maximum = str(maximum/cnt*100)[:5]
    fig1 = plt.figure()
    plt.title('incr@'+str(incr)+'%; purge '+prg+'%; max '+maximum+'% @0.'+bestThr)
    plt.xticks([], [])
    plt.xlabel('Species delim. range (%): ' + str(lower) +  ' to ' + str(upper))
    plt.plot(props)
    fig1.savefig(f + '.pdf', dpi = 900)

if __name__ == '__main__':
    main()
