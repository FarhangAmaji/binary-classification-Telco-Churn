import time
import threading

ti=time.time

def q(*t,typ="a",filewrite="",path="",s="",il="",noPrint=0,filename='plog'):
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

def inputTimeout(prompt, timeout=30):
    print(prompt)
    user_input = [None]

    def inputThread():
        user_input[0] = input()

    # Start the input thread
    thread = threading.Thread(target=inputThread)
    thread.start()

    # Wait for the thread to finish or timeout
    thread.join(timeout)

    # Check if input was received or timeout occurred
    if thread.is_alive():
        # print("No input received. Continuing the code...")
        return False
    else:
        return user_input[0]
