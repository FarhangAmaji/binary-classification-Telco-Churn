import time
def q(*t,typ="a",filewrite="",path="",s="",il="",noPrint=0,filename='plog'):#kkk some options like filename exist in original print#kkk rewrite it
    '''
    Parameters
    ----------
    s : TYPE, optional
        0 gets input, anything except ENTER for answer would terminate the code.
        1 always will terminate the code.
    il : TYPE, optional
        separate lines. The default is "" is for non separate lines but anything else will separate it.
    '''
    if path:
        pastpath=os.getcwd()
        os.chdir(fr'{path}')
    if filewrite:
        writestring=""
        for i in t:
            writestring+=str(i)+" "
        with open(f'{filename}.txt',f'{typ}', encoding='utf-8') as fh:#'w' 'a'
            fh.write(f"{writestring}\n")
    if il=="":
        if not noPrint:
            print(*t)
    else:
        if not noPrint:
            for t_ in t:
                print(t_)
    if path!="":
        os.chdir(fr'{pastpath}')
    if s==0:
        inp=input("enter something to abrupt")
        if inp!="":
            1/0
    if s==1:
        1/0

ti=time.time