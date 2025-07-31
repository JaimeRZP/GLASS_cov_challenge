"""
  ispice defines tools to run Spice from Python,
  either in the Planck HFI DMC (aka piolib, objects managed by database)
  or using FITS files
  
Includes:
  Main routine:
        ispice()
  Support routines:
        flatten()
        is_fits()
        look_for_file()
        thinc_exists()
        tools()
        tf2yn()
        subreadme()
        readme()

HISTORY:
     2010-10: first version in python2
     2015-10: introduction in public release
     2019-06: supports python3
     2020-01: supports True/False flags besides "YES"/"NO";
       access to HTML documentation
     2020-05: added support for omp_num_threads, 
         kernelsfileout, noiseclfile, noisecorfile, normfac
     2020-09: added alm?fileout
     2020-10: accept $ and ~ in file names ; use subprocess to run jobs
     2021-04: avoid error when $OMP_NUM_THREADS not defined
     2021-06: supports $CONDA_PREFIX, add alm?weightin
     2023-04: accepts cl_inmap_file (precomputed C(l) of masked map) instead of input raw map;
              verbosity can be set in {0,1,2}; ispice.readme() now works
     2024-09: compatible with python3.12
  """

import os
from os.path import dirname
# default value of omp_num_threads
ompntdef = os.getenv("OMP_NUM_THREADS") # read from environmment, default to None
if (ompntdef is None):
    try:
        import multiprocessing
        ompntdef=multiprocessing.cpu_count()
    except:
        ompntdef=1
readmehtml = "http://www2.iap.fr/users/hivon/software/PolSpice/README.html"
conda_prefix=os.getenv("CONDA_PREFIX")
__version__ = "3.8.3"



def flatten(x):
    """flatten():  turns nested list or tuple into single list"""
    result = []
    if isinstance(x, str):
        result = x
    else:
        for el in x:
            if hasattr(el, "__iter__") and not isinstance(el, str):
                result.extend(flatten(el))
            else:
                result.append(el)
    return result

def is_fits(inlist):
    """test presence of FITS file in a list"""
    import os
    flatlist = flatten(inlist)
    if isinstance(flatlist, str): flatlist = [ flatlist ]
    isFits = False
    for f in flatlist:
        if isinstance(f, str):
            ff = os.path.expandvars(os.path.expanduser(f))#expand $ and ~
            if os.path.exists(ff):
                isFits = os.path.isfile(ff)
    return isFits

def thinc_exists():
    """thinc_exists():  test existence of thinc module (DMC specific)"""
    try:
        import importlib
        importlib.import_module("thinc")
        importlib.import_module("piolib")
        myDMC = True
    except ImportError:
        myDMC = False
    return myDMC

def look_for_file(lookfor = "spice", extra=None, crash_if_not_found=False):
    """look for spice code in neighbouring directories"""
    from os      import walk
    from os.path import join, dirname

    bdir = dirname(__file__)
    locs = [join(bdir,"."), join(bdir,"..")]
    if (extra is not None and extra !=""): locs.append(extra)
    path = ""
    for loc in locs:
        for root, dirs, files in walk(loc):
            if lookfor in files:
                path = join(root, lookfor)
                break
        if (path != ""): break
    if (crash_if_not_found and path == ""):
        import sys
        sys.exit(
            "Error: code (%s) not found in %s !\n\tUse binpath to specify full path to code."%(
            lookfor,",".join(locs)))
    return path


def tf2yn(arg):
    """Turn True/False into "YES"/"NO" """
    out = arg
    if (type(arg) == bool):
        out = "YES" if (arg) else "NO"
    return out

def subreadme(f=readmehtml):
    try:
        import webbrowser
        webbrowser.open_new_tab(f)
    except:
        print("The web browser could not be opened.")
        print("See %s for PolSpice documentation,"%(readmehtml))
        print("or type    spice -help  ")
    return

def readme():
    locrh = look_for_file("README.html")
    if (locrh != ""):
        try:
            subreadme(f="file://%s"%(locrh))
        except:
            subreadme()
    else:
        subreadme()
    return
readme.__doc__="""
    readme() opens a web browser tab with the PolSpice documentation README.html
    found in {0}
    or in    {1}
""".format(dirname(dirname(__file__)),readmehtml)

#----------------------------------------------------------------

class tools(object):
    """tools() defines context dependent routines"""
    def __init__(self, object):
        self.type = object
        in_thinc = thinc_exists()
        in_fits  = is_fits(object)
        if (not in_thinc and not in_fits):
            import sys
            print("Input files do not exist or are not in FITS"%(object))
            #print("Input files (%s) do not exist or are not in FITS"%(object))
            #print("Input files are not in FITS, and DMC not available")
            sys.exit("Aborting")
            
        self.inDMC = in_thinc and not in_fits
        self.command = ""
    # -------------------------------------------
    def myNewJob(self, code, label="", cast=False):
        from os.path import abspath
        if (self.inDMC):
            from thinc import NewJob
            return NewJob(code, label=label, cast=cast)
        else:
            if (code == "spice" or code == "spice_SP" or code == "spice_DP"):
                mycom = look_for_file(code, extra=conda_prefix, crash_if_not_found=True)
            else:
                mycom = code
            self.command += abspath(mycom)+" "
            return 0

    def mySet(self, jobid, name, value_):
        value = tf2yn(value_) # turn True/False into "YES"/"NO"
        if (self.inDMC):
            from thinc import Set
            Set(jobid, name, value)
        else:
            #print str(name)+" = "+str(value)
            self.command += "-%s %s "%(str(name),str(value))

    def mySubmit(self, jobid, exportEnviron=None):
        """mySubmit(jobid, exportEnviron=None)
            runs a job in shell with customized environment and prints stdout and stderr."""
        if (self.inDMC):
            from thinc import Submit
            Submit(jobid, exportEnviron=exportEnviron)
        else:
            import os
            # print("Submitting job %s"%(str(jobid)))
            # memory foot-print: Linux: /usr/bin/time -f"Max memory: %M kbytes", MacOS: ????
#             precommand=""
#             if isinstance(exportEnviron, dict):
#                 for k,v in exportEnviron.items():
#                     precommand += "%s=%s "%(k,v)
#             self.command = precommand + self.command
#             print(self.command)
#             os.system(self.command)
            import subprocess # 2020-10-30
            # self.command += "  & "
            print(self.command)
            print(subprocess.check_output(
                self.command,shell=True, # Run spice in a shell and print stdout to the Python session
                universal_newlines=True, # interpret the output as text and encode it properly
                env=dict(os.environ, **exportEnviron), # customize environment
                stderr=subprocess.STDOUT))# make sure that stderr is also printed
    # -------------------------------------------
    def inDMC(self):
        return self.inDMC
    
    def parse_mapfile(self, jobid, mapfile, case):
        import numpy as np
        """parse_mapfile(jobid, mapfile, case)"""
        if (self.inDMC):
            keywords=["mapfile","mapQfile","mapUfile"]
        else:
            keywords=["mapfile"]
        if (case==2): keywords = [        k+"2" for k in keywords]
        if (case==3): keywords = ["extra"+k     for k in keywords]
        if (case==4): keywords = ["extra"+k+"2" for k in keywords]
    
        ##print case, mapfile,type(mapfile)
        if isinstance(mapfile, (list,tuple)): # list
            if (isinstance(mapfile[0], (list,tuple))  and (case==1 or case==2) ): # list of list
                ##print "==>List of list"
                if (self.inDMC):
                    if (case==1):kw = ["listmapweights1_",["listmapfiles1_","listmapfilesQ1_","listmapfilesU1_"]]
                    if (case==2):kw = ["listmapweights2_",["listmapfiles2_","listmapfilesQ2_","listmapfilesU2_"]]
                else:
                    if (case==1):kw = ["listmapweights1_",["listmapfiles1_"]]
                    if (case==2):kw = ["listmapweights2_",["listmapfiles2_"]]
                for i in range(len(mapfile)):
                    si=str(i+1)
                    if isinstance(mapfile[i][0],(type(1),type(1.),np.float32,np.float64)):
                        self.mySet(jobid, "%s%s"%(kw[0],si), mapfile[i][0])
                    else:
                        print("Warning: number was expected, got "+str(mapfile[i][0]))
                    if isinstance(mapfile[i][1], (list,tuple)):
                        nl = len(mapfile[i][1])
                        if (nl != 1 and nl != 3):
                            print("Warning: expected a 1 or 3-element list, found "+str(nl))
                            print(mapfile[i][1])
                        for j in range(nl):
                            self.mySet(jobid, "%s%s"%(kw[1][j],si), mapfile[i][1][j])
                    else:
                        self.mySet(jobid, "%s%s"%(kw[1][j],si), mapfile[i][1])
            else: # simple list
                ##print "==>List"
                if (len(mapfile[0])>0): # non-empty string in list
                    self.mySet(jobid, keywords[0], mapfile[0])
                    if (len(mapfile) == 3):# 3 element list
                        self.mySet(jobid, keywords[1], mapfile[1])
                        self.mySet(jobid, keywords[2], mapfile[2])
        else: #  not list -> string
            ##print "==>String"
            if (len(mapfile)>0): # non-empty string
                self.mySet(jobid, keywords[0],     mapfile)
    

# =======================================================================================
            
def ispice(
    mapin1, clout, nlmax=-1,
    alm1fileout="",
    alm2fileout="",
    alm1weightin="",
    alm2weightin="",
    apodizetype=0,
    apodizesigma="NO",
    beam1="NO", beam2="NO",
    beam_file1="", beam_file2="",
    binpath="",
    cl_inmap_file="",
    cl_inmask_file="",
    cl_outmap_file="",
    cl_outmask_file="",
    corfile="",
    covfileout="",
    decouple="NO",
    extramapfile1="", extramapfile2="",
    fits_out="YES",
    kernelsfileout="",
    mapfile2="",
    maskfile1="", maskfile2="", maskfilep1="", maskfilep2="",
    noiseclfile="",
    noisecorfile="",\
    normfac=1.0,    
    omp_num_threads=ompntdef,
    pixelfile="YES", pixelfile2="YES", polarization="NO",
    subav="NO",
    subdipole="NO",
    symmetric_cl="NO",
    tenormfilein="",tenormfileout="",
    tf_file="",
    thetamax="NO" ,
    tolerance="NO",
    verbosity=2,
    weightfile1="", weightfilep1="",
    weightfile2="", weightfilep2="",
    weightpower1=1.0, weightpower2=1.0, weightpowerp1=1.0, weightpowerp2=1.0,
    windowfilein="",windowfileout="",
    label="spice", submit=None):

    if len(cl_inmap_file.strip()) > 0:
        mytools = tools( cl_inmap_file )
    else:
        mytools = tools( mapin1 )

    command="spice"
    if (not mytools.inDMC and len(binpath)>0):
        command=binpath
    myJob = mytools.myNewJob(command, label = label, cast = True)
    
    if (len(alm1fileout)>0):
        mytools.mySet(myJob, "alm1fileout", alm1fileout)
    if (len(alm2fileout)>0):
        mytools.mySet(myJob, "alm2fileout", alm2fileout)
    if (len(alm1weightin)>0):
        mytools.mySet(myJob, "alm1weightin", alm1weightin)
    if (len(alm2weightin)>0):
        mytools.mySet(myJob, "alm2weightin", alm2weightin)

    mytools.mySet(myJob, "apodizesigma", apodizesigma)
    mytools.mySet(myJob, "apodizetype",  apodizetype)
    
    mytools.mySet(myJob, "beam",         beam1)
    if (len(beam_file1)>0):
        mytools.mySet(myJob, "beam_file",    beam_file1)
    mytools.mySet(myJob, "beam2",        beam2)
    if (len(beam_file2)>0):
        mytools.mySet(myJob, "beam_file2",   beam_file2)
        
    mytools.mySet(myJob, "clfile",       clout)

    if (len(cl_inmap_file)>0):
        mytools.mySet(myJob, "cl_inmap_file",  cl_inmap_file)
    if (len(cl_inmask_file)>0):
        mytools.mySet(myJob, "cl_inmask_file",  cl_inmask_file)
    if (len(cl_outmap_file)>0):
        mytools.mySet(myJob, "cl_outmap_file",  cl_outmap_file)
    if (len(cl_outmask_file)>0):
        mytools.mySet(myJob, "cl_outmask_file",  cl_outmask_file)

    if (len(covfileout)>0):
        mytools.mySet(myJob, "covfileout",   covfileout)
    if (len(corfile)>0):
        mytools.mySet(myJob, "corfile",  corfile)
    mytools.mySet(myJob, "decouple",     decouple)

    mytools.parse_mapfile(myJob, mapin1,        1)
    mytools.parse_mapfile(myJob, mapfile2,      2)
    mytools.parse_mapfile(myJob, extramapfile1, 3)
    mytools.parse_mapfile(myJob, extramapfile2, 4)

    if (not mytools.inDMC):
        mytools.mySet(myJob, "fits_out", fits_out)
        
    if (len(kernelsfileout)>0):
        mytools.mySet(myJob, "kernelsfileout", kernelsfileout)

    if (len(maskfile1)>0):
        mytools.mySet(myJob, "maskfile",      maskfile1)
    if (len(maskfilep1)>0):
        mytools.mySet(myJob, "maskfilep",     maskfilep1)
    if (len(maskfile2)>0):
        mytools.mySet(myJob, "maskfile2",     maskfile2)
    if (len(maskfilep2)>0):
        mytools.mySet(myJob, "maskfilep2",    maskfilep2)
        
    if (len(weightfile1)>0):
        mytools.mySet(myJob, "weightfile",      weightfile1)
    if (len(weightfilep1)>0):
        mytools.mySet(myJob, "weightfilep",     weightfilep1)
    if (len(weightfile2)>0):
        mytools.mySet(myJob, "weightfile2",     weightfile2)
    if (len(weightfilep2)>0):
        mytools.mySet(myJob, "weightfilep2",    weightfilep2)
        
    mytools.mySet(myJob, "nlmax",        nlmax)
    if (len(noiseclfile)>0):
        mytools.mySet(myJob, "noiseclfile",    noiseclfile)
    if (len(noisecorfile)>0):
        mytools.mySet(myJob, "noisecorfile",    noisecorfile)

    #mytools.mySet(myJob, "normfac",      "1.00000")
    mytools.mySet(myJob, "normfac",   normfac)
    mytools.mySet(myJob, "npairsthreshold", "0.00000")
    mytools.mySet(myJob, "overwrite",       "YES")
    mytools.mySet(myJob, "polarization",    polarization)
    mytools.mySet(myJob, "pixelfile",       pixelfile)
    mytools.mySet(myJob, "pixelfile2",      pixelfile2)
    mytools.mySet(myJob, "subav",           subav)
    mytools.mySet(myJob, "subdipole",       subdipole)
    mytools.mySet(myJob, "symmetric_cl",    symmetric_cl)
    if (len(tenormfilein)>0):
        mytools.mySet(myJob, "tenormfilein", tenormfilein)
    if (len(tenormfileout)>0):
        mytools.mySet(myJob, "tenormfileout",tenormfileout)
    if (len(tf_file)>0):
        mytools.mySet(myJob, "tf_file",     tf_file)
    mytools.mySet(myJob, "thetamax",        thetamax)
    mytools.mySet(myJob, "tolerance",       tolerance)
    mytools.mySet(myJob, "verbosity",       verbosity)
    mytools.mySet(myJob, "weightpower",     weightpower1)
    mytools.mySet(myJob, "weightpower2",    weightpower2)
    mytools.mySet(myJob, "weightpowerp",    weightpowerp1)
    mytools.mySet(myJob, "weightpowerp2",   weightpowerp2)
    if (len(windowfilein)>0):
        mytools.mySet(myJob, "windowfilein",    windowfilein)
    if (len(windowfileout)>0):
        mytools.mySet(myJob, "windowfileout",   windowfileout)

    if (mytools.inDMC):
        mytools.mySet(myJob, "pbs_extraOption","-l place=scatter:excl -l select=1:ncpus="+str(omp_num_threads))


    mysubmit=submit
    if (mysubmit==None):
        if (mytools.inDMC):
            mysubmit=False # default: do not submit in DMC
        else:
            mysubmit=True # default: do submit in FITS

    if mysubmit:
        exev = {"OMP_NUM_THREADS":str(ompntdef)} # default value read from environment
        if int(omp_num_threads)>0: exev={"OMP_NUM_THREADS":str(omp_num_threads)} # user provided value
        mytools.mySubmit(myJob, exportEnviron=exev)
    else:
        if (not mytools.inDMC):
            print(mytools.command)
            print 
            print("The command above was NOT RUN.")
            print("Select submit=True in ispice to run it, or copy and paste it in a terminal")
    return myJob

ispice.__doc__ =     """
        ispice(mapin1, clout, nlmax=-1,
           alm1fileout="",
           alm2fileout="",
           alm1weightin="",
           alm2weightin="",
           apodizetype=0, 
           apodizesigma="NO", 
           beam1="NO", beam2="NO", 
           beam_file1="", beam_file2="",
           binpath="",
           cl_inmap_file="", 
           cl_inmask_file="", 
           cl_outmap_file="", 
           cl_outmask_file="", 
           covfileout="", 
           corfile="", 
           decouple="NO", 
           extramapfile1="", extramapfile2="", 
           fits_out="YES", 
           kernelsfileout="", 
           mapfile2="", 
           maskfile1="", maskfile2="", maskfilep1="", maskfilep2="",
           noiseclfile="", 
           noisecorfile="", 
           normfac="", 
           omp_num_threads={0},
           pixelfile="YES", pixelfile2="YES", polarization="NO", 
           subav="NO", 
           subdipole="NO", 
           symmetric_cl="NO",
           tenormfilein="", tenormfileout="",
           tf_file="", 
           thetamax="NO" ,
           tolerance="NO" ,
           verbosity=2,
           weightfile1="", weightfilep1="", 
           weightfile2="", weightfilep2="", 
           weightpower1=1.0, weightpower2=1.0, weightpowerp1=1.0, weightpowerp2=1.0, 
           windowfilein="",windowfileout="", 
           label="spice", submit=None):

           Python3 interface to F90 spice code ({4})

           Required:
             mapin1:  input I map, or list of [I,Q,U] maps,  DMC objects of type MAP
                      input I or IQU map,                    FITS files
                     or list of lists for on-the-fly weighted linear combination (LC) of maps:
                     eg:  [ [w1, I1 ],       [w2,  I2 ],       [w3,  I3 ],      ... ]
                     or   [ [w1,[I1]],       [w2, [I2]],       [w3, [I3]],      ... ]
                     or   [ [w1,[IQU1]],     [w2, [IQU2]],     [w3, [IQU3]]     ... ] (FITS only)
                     or   [ [w1,[I1,Q1,U1]], [w2, [I2,Q2,U2]], [w3, [I3,Q3,U3]] ... ] (DMC only)
                     so that mapin1_I = w1*I1 + w2*I2 + w3*I3 + ...
                       (and  mapin1_Q = w1*Q1 + w2*Q2 + w3*Q3 + ...)
                     where w* is a scalar number and I*,Q*,U* are MAPtype objects
                     
             clout:    output C(l) (either auto or cross),   object  of type CL
           
           Optional:
             alm1fileout:   see [1] below
             alm2fileout:   see [1] 
             alm1weightin:  see [1] 
             alm2weightin:  see [1] 
             apodizesigma:  see [1]
             apodizetype:   see [1]
             beam1, beam2: beam(s) FWHM in arcmin
             beam_file*: input B(l) file(s),                 FITS files or objects of type CL
             binpath: name of Spice executable, or its relative or absolute path.
                If set to "spice" (=default value), "spice_SP" or "Spice_DP",
                 a code of the same name will be looked-for in the directory ({1})
                 containing this python script, or in its sibling directories,
                 or in $CONDA_PREFIX (={3}).
                Paths starting with "." are relative to current directory.
                Those starting with "/" are absolute.
             cl_*_file:     see [1]
             covfileout: output C(l)C(l) covariance matrix,  FITS file  or object  of type TAB3D
             corfile:       see [1]
             decouple:      see [1]
             extramapfile1: I or [I,Q,U] map to be added to mapin1
                will be ignored by Spice if mapin1 is a LC
             extramapfile2: I or [I,Q,U] map to be added to mapfile2
                will be ignored by Spice if mapfile2 is a LC
             fits_out: output files are in FITS instead of plain ASCII
             kernelsfileout: see [1]
             label: name given to job
             mapfile2: map(s) or LC of maps for cross-spectrum
                see mapin1 for format
             maskfile*:     see [1]
             noiseclfile:   see [1] 
             noisecorfile:  see [1] 
             normfac:       see [1] 
             omp_num_threads: number of OMP threads to be used (temporarily (re)defining OMP_NUM_THREADS if >0)
                              default value ({0}): current value of $OMP_NUM_THREADS, or number of virtual CPUs, or 1.
             pixelfile*:    see [1]
             polarization:  see [1]
             subav:         see [1]
             subdipole:     see [1]
             submit:  if set to False, the command string is generated, but job is not submitted
             symmetric_cl:  see [1]
             tenormfile*:   see [1]
             tf_file:       see [1]
             tolerance:     see [1]
             verbosity:     see [1]
             weightfile*:   see [1]
             weightpower*:  see [1]
             windowfile*:   see [1]
           
           [1] ispice.readme()
               or
               {2}

           Note: values "YES" and "NO" can be replaced by True and False respectively,
               wherever applicable.


           Examples:
           
           import ispice
           
           # compute polarized spectrum of mapfile:
           ispice.ispice(mapfile, clfile, polarization=True, pixelfile=True, subdipole=True)
           
           # help on ispice (python interface)
           help(ispice.ispice)

           # help on PolSpice (F90 backend):
           ispice.readme()
           
           
    """.format(str(ompntdef),dirname(__file__),readmehtml,str(conda_prefix),str(__version__))
