%
% ``tcvg'' LaTeX class.
% - modification of the acmsiggraph.cls class
% - last modifications 2004/04/06 Torsten Moeller
%    * many modifications to conform to TCVG standard
%      - new heading style
%      - new caption style
%      - new reference style
%      - copyrightspace reduced to 0.5in

% - borrows *heavily* from Daniel Kartch's 'siggraph' class. 
% - also uses pieces of 'apalike.sty' and 'authordate1-4.cls'
% - version 0.1 2001/06/01 Stephen Spencer (spencer@acm.org)
% - version 0.2 2001/10/15 Stephen Spencer
%   - the "\onlineid" variable works with the "review" mode, placing a
%     banner across the top of each page "Online Submission ID 'onlineid'
% - version 0.3 2002/01/11 Stephen Spencer
%   - disabled the cover page option.
% - version 0.4 2002/01/23 Stephen Spencer
%   - based on suggestions from James O'Brien, the following changes:
%     - correction if '\ifcamera' and '\ifcameraelse' commands
%     - page numbering in both review and preprint modes
%     have been made.

% ------------ identification --------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tcvg}[2004/04/04 IEEE TCVG]

% ------------ initial code --------------

\newif\ifacm@camera
\newif\ifacm@preprint
\newif\ifacm@review
\newif\ifacm@doublespaced

\newcommand{\acm@columnmode}{}
\newcommand{\acm@pointsize}{}

% ------------ declaration of options --------------

% 'cameraready' option.

\DeclareOption{cameraready}{%
    \acm@cameratrue%
    \acm@preprintfalse%
    \acm@reviewfalse%
    \renewcommand{\acm@columnmode}{twocolumn}%
    \acm@doublespacedfalse%
    \renewcommand{\acm@pointsize}{10.5pt}}

% 'review' option.

\DeclareOption{review}{%
    \acm@camerafalse%
    \acm@preprintfalse%
    \acm@reviewtrue%
    \renewcommand{\acm@columnmode}{twocolumn}%
    \acm@doublespacedfalse%
    \renewcommand{\acm@pointsize}{10pt}}

% 'widereview' option.

\DeclareOption{widereview}{%
    \acm@camerafalse%
    \acm@preprintfalse%
    \acm@reviewtrue%
    \renewcommand{\acm@columnmode}{onecolumn}%
    \acm@doublespacedtrue%
    \renewcommand{\acm@pointsize}{12pt}}

% 'preprint' option.

\DeclareOption{preprint}{%
    \acm@camerafalse%
    \acm@preprinttrue%
    \acm@reviewfalse%
    \renewcommand{\acm@columnmode}{twocolumn}%
    \acm@doublespacedfalse%
    \renewcommand{\acm@pointsize}{9pt}}

% Assume, for the moment, that we're preparing a 'cameraready' version.

\ExecuteOptions{cameraready}

% Here's a warning command for use just below.

\newcommand{\acm@optwarning}[1]{%
    \ifacm@camera
        \ClassWarningNoLine{tcvg}%
            {#1 option used in camera-ready mode.\MessageBreak
             This violates submission specifications}
    \fi
}

% The 'onecolumn' option doesn't work in 'cameraready' mode.

\DeclareOption{onecolumn}{%
    \acm@optwarning{onecolumn}
    \renewcommand{\acm@columnmode}{onecolumn}}

% The 'twocolumn' option works in 'cameraready' mode.

\DeclareOption{twocolumn}{%
    \renewcommand{\acm@columnmode}{twocolumn}}

% Only the '9pt' size works in 'cameraready' mode.

\DeclareOption{9pt}{%
    \renewcommand{\acm@pointsize}{9pt}}
\DeclareOption{10pt}{%
    \acm@optwarning{10pt}
    \renewcommand{\acm@pointsize}{10pt}}
\DeclareOption{11pt}{%
    \acm@optwarning{11pt}
    \renewcommand{\acm@pointsize}{11pt}}
\DeclareOption{12pt}{%
    \acm@optwarning{12pt}
    \renewcommand{\acm@pointsize}{12pt}}

% The 'singlespace' option works in 'cameraready' mode.

\DeclareOption{singlespace}{%
    \acm@doublespacedfalse}

% The 'doublespace' option does not work in 'cameraready' mode.

\DeclareOption{doublespace}{%
    \acm@optwarning{doublespace}
    \acm@doublespacedtrue}

% No 'titlepage' option in 'cameraready' mode.

\DeclareOption{titlepage}{%
    \OptionNotUsed%
    \ClassWarningNoLine{tcvg}{titlepage option not allowed}}

% No 'landscape' mode in 'cameraready' mode, either.

\DeclareOption{landscape}{%
    \OptionNotUsed%
    \ClassWarningNoLine{tcvg}{landscape option not allowed}}

% Pass everything else to the 'article' class, upon which this is based.

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\PassOptionsToClass{\acm@columnmode}{article}
\ifdim\acm@pointsize>9pt
    \PassOptionsToClass{\acm@pointsize}{article}
\fi

% ------------ package loading --------------

\LoadClass{article}

% ------------ main code --------------

\newcommand{\acm@onlineid}{}
\newcommand{\onlineid}[1]{\renewcommand{\acm@onlineid}{#1}}

\newcommand{\acm@preprinttext}{To appear in IEEE Visualization 2004 conference proceedings.}
\newcommand{\preprinttext}[1]{\renewcommand{\acm@preprinttext}{#1}}

\newcommand{\acm@reviewtext}{Online Submission ID: \acm@onlineid}
\newcommand{\reviewtext}[1]{\renewcommand{\acm@reviewtext}{#1}}

\newcommand{\ifcamera}[1]{\ifacm@camera #1 \fi}
\newcommand{\ifreview}[1]{\ifacm@review #1 \fi}
\newcommand{\ifcameraelse}[2]{\ifacm@camera #1 \else #2 \fi}
\newcommand{\ifreviewelse}[2]{\ifacm@review #1 \else #2 \fi}
\setlength{\textheight}{9.75in}
\setlength{\topmargin}{-1.0in}
\setlength{\headheight}{0.1in}
\setlength{\headsep}{0.250in}
\setlength{\footskip}{0.25in}
\flushbottom
\setlength{\textwidth}{7.25in}
\setlength{\oddsidemargin}{-0.25in}
\setlength{\evensidemargin}{-0.25in}
\setlength{\columnsep}{1.5pc}
%%\setlength{\parindent}{1em}

\renewcommand{\ps@plain}%
    {%
    \renewcommand{\@oddhead}{}%
    \renewcommand{\@oddfoot}{}%

    \ifacm@preprint%
      \renewcommand{\@oddhead}{\hfil\textit{\large\acm@preprinttext}\hfil}%
      \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
    \fi%

    \ifacm@review%
      \renewcommand{\@oddhead}{\hfil\textit{\large\acm@reviewtext}\hfil}%
      \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
    \fi%

    \renewcommand{\@evenhead}{\@oddhead}%
    \renewcommand{\@evenfoot}{\@oddfoot}%
    }

\renewcommand{\ps@empty}%
    {%
    \renewcommand{\@oddhead}{}%
    \renewcommand{\@oddfoot}{}%

    \ifacm@preprint%
      \renewcommand{\@oddhead}{\hfil\textit{\large\acm@preprinttext}\hfil}%
      \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
    \fi%

    \ifacm@review%
      \renewcommand{\@oddhead}{\hfil\textit{\large\acm@reviewtext}\hfil}%
      \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
    \fi%

    \renewcommand{\@evenhead}{\@oddhead}%
    \renewcommand{\@evenfoot}{\@oddfoot}%
    }

% no page numbers - they are added in production

\pagestyle{plain}

\newcommand{\acm@setninepoint}{
    \renewcommand\normalsize{%
        \@setfontsize\normalsize\@ixpt\@xpt
        \abovedisplayskip 9\p@ \@plus2\p@ \@minus4\p@
        \abovedisplayshortskip \z@ \@plus3\p@
        \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
        \belowdisplayskip \abovedisplayskip
        \let\@listi\@listI}
    \renewcommand\small{%
        \@setfontsize\small\@viipt\@ixpt
        \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
        \abovedisplayshortskip \z@ \@plus2\p@
        \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
        \def\@listi{\leftmargin\leftmargini
                    \topsep 4\p@ \@plus2\p@ \@minus2\p@
                    \parsep 2\p@ \@plus\p@ \@minus\p@
                    \itemsep \parsep}%
        \belowdisplayskip \abovedisplayskip}
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize\@viiipt{9.5}%
        \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
        \abovedisplayshortskip \z@ \@plus\p@
        \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
        \def\@listi{\leftmargin\leftmargini
                    \topsep 3\p@ \@plus\p@ \@minus\p@
                    \parsep 2\p@ \@plus\p@ \@minus\p@
                    \itemsep \parsep}%
        \belowdisplayskip \abovedisplayskip}
    \renewcommand\scriptsize{\@setfontsize\scriptsize\@viiipt{9.5}}
    \renewcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
    \renewcommand\large{\@setfontsize\large\@xpt\@xiipt}
    \renewcommand\Large{\@setfontsize\Large\@xiipt{12}}
    \renewcommand\LARGE{\@setfontsize\LARGE\@xivpt{16}}
    \renewcommand\huge{\@setfontsize\huge\@xviipt{20}}
    \renewcommand\Huge{\@setfontsize\Huge\@xxpt{22}}
    \selectfont
}
\ifdim\acm@pointsize=9pt
    \acm@setninepoint
\fi

\newcommand{\acm@sectionfont}{}
\newcommand{\sectionfont}[1]{\renewcommand{\acm@sectionfont}{#1}}
\renewcommand\section{\@startsection{section}{1}{\z@}%
                {-1.5ex \@plus -1ex \@minus -.2ex}%
                {2.3ex \@plus.2ex}%
                {\reset@font\large\bfseries\scshape\acm@sectionfont}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                {-3.25ex\@plus -1ex \@minus -.2ex}%
                {1.5ex \@plus .2ex}%
                {\reset@font\normalsize\bfseries\acm@sectionfont}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                {-3.25ex\@plus -1ex \@minus -.2ex}%
                {1.5ex \@plus .2ex}%
                {\reset@font\normalsize\acm@sectionfont}}

\renewcommand\paragraph{\@startsection{paragraph}{4}{1em}%
                {1ex \@plus1ex \@minus.2ex}%
                {-1em}%
                {\reset@font\normalsize\sffamily\bfseries\acm@sectionfont}}
%\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
%                {3.25ex \@plus1ex \@minus.2ex}%
%                {-1em}%
%                {\reset@font\normalsize\sffamily\bfseries\acm@sectionfont}}

\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                {3.25ex \@plus1ex \@minus .2ex}%
                {-1em}%
                {\reset@font\normalsize\sffamily\bfseries\acm@sectionfont}}

%% make captionfont 8pt
\newcommand{\captionfonts}{\small\sffamily}

\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\captionfonts #1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    {\captionfonts #1: #2\par}
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  %\vskip\belowcaptionskip}
}

%% fix the font size of the bibliography to 8pt
\newdimen\bibindent
\newdimen\bibspacing
\setlength\bibindent{1pt}
\setlength{\bibspacing}{\z@}
\renewenvironment{thebibliography}[1]
     {\section*{\refname}%
      \small%
      \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \itemsep\bibspacing    % should this be commented out?
            \parsep\bibspacing     % should this be commented out?
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}


\newcommand{\acm@empty}{}

\newcommand{\acm@affiliation}{}
\newcommand{\affiliation}[1]{\renewcommand{\acm@affiliation}{#1}}

\newcommand{\acm@category}{}
\newcommand{\category}[1]{\renewcommand{\acm@category}{#1}}

\newcommand{\acmcategory}[1]{\category{#1}}
\newcommand{\acm@format}{}

\newcommand{\format}[1]{\renewcommand{\acm@format}{#1}}
\newcommand{\acmformat}[1]{\format{#1}}

\newcommand{\acm@teaser}{}
\newcommand{\teaser}[1]{\renewcommand{\acm@teaser}{#1}}

\newlength{\titlespace}
\setlength{\titlespace}{0.25in}

\newlength{\teaserspace}
\setlength{\teaserspace}{0.25in}

\ifacm@review
    \renewcommand{\@maketitle}{%
        \begin{center}%
            \renewcommand{\thanks}[1]{}
            {\LARGE\sffamily\bfseries\acm@sectionfont \@title \par}%
%            \vspace{1\baselineskip}%
            {Category: \acm@category \par}%
%            \vspace{0.25\baselineskip}%        % no longer needed (1996)
%            {Format: \acm@format \par}%        % no longer needed (1996)
%            \vspace{\titlespace}%
            \ifx\acm@teaser\acm@empty \else%
                \begingroup%
                    \def\@captype{figure}%
                    \acm@teaser%
                \endgroup\par%
            \fi%
        \end{center} \par%
    }
\else
    \renewcommand{\@maketitle}{%
        \begin{center}%
            {\LARGE\sffamily\bfseries\acm@sectionfont \@title \par}%
%            \vspace{1\baselineskip}%
            \large\sffamily\acm@sectionfont 
            \begin{tabular}[t]{c}%
                \@author
            \end{tabular}\par%
            \ifx\acm@affiliation\acm@empty \else%
                \vspace{1\baselineskip}%
                \acm@affiliation\par%
            \fi%
%            \vspace{\titlespace}%
            \ifx\acm@teaser\acm@empty \else%
                \begingroup%
                    \def\@captype{figure}%
                    \acm@teaser%
                \endgroup\par%
            \fi%
        \end{center} \par%
    }
\fi

\let\acm@origmaketitle\maketitle
\let\acm@origand\and
\renewcommand{\maketitle}{%
    \let\acm@title\@title%
    \let\acm@author\@author%
    \acm@origmaketitle%
    \thispagestyle{empty}%
    \ifacm@doublespaced%
        \renewcommand{\baselinestretch}{1.66}\selectfont%
    \fi
}

\newtoks\acm@abs
\ifacm@review
    \long\def\acm@add#1{\global\acm@abs\expandafter{\the\acm@abs#1}}
    \long\def\acm@collect{%
        \global\acm@abs{}%
        \let\abstract\acm@@collect
        \abstract
    }
    \long\def\acm@@collect#1\end#2{%
        \def\@tempa{#2}%
        \ifx\@tempa\@currenvir
            \acm@add{#1}%
            \edef\abstract{\noexpand\end{\@tempa}}%
        \else
            \acm@add{#1\end{#2}}%
        \fi
        \abstract
    }
    \newcommand{\acm@modify}{%
        \let\acm@origabs\abstract%
        \let\acm@origendabs\endabstract%
        \renewenvironment{abstract}%
            {\acm@collect}%
            {\begingroup
                \let\abstract\acm@origabs
                \let\endabstract\acm@origendabs
                \begin{abstract} \the\acm@abs \end{abstract}
            \endgroup}%
    }
    \AtBeginDocument{\acm@modify}
\fi
\newcommand{\keywordlist}{%
    \vspace{0.5\baselineskip}%
    \par\noindent \textbf{Keywords: } \acm@keywords%
}
\newenvironment{CRcatlist}{%
    \vspace{0.5\baselineskip}%
    \par\noindent \textbf{CR Categories: }%
}{}

\newcommand{\CRcat}[4]{%
    #1 [#2]%
    \ifx#3\acm@empty \else : #3\fi%
    \ifx#4\acm@empty \else ---#4\fi%
}

% leave a 0.5 inch space at the bottom of the left column
% on the first page for the copyright block.

\newlength{\acm@copyrightlength}
\setlength{\acm@copyrightlength}{0.5in}

\newcommand{\copyrightspace}{%
    \renewcommand{\thefootnote}{}%
    \footnotetext[0]{\rule[\acm@copyrightlength]{2.71828in}{0in}}%
    \renewcommand{\thefootnote}{\arabic{footnote}}
}

\renewcommand{\footnoterule}{%
    \kern-3pt
    \hrule width 0.45\columnwidth
    \kern 2.6pt}

\newcommand{\acm@contactname}{}
\newcommand{\contactname}[1]{\renewcommand{\acm@contactname}{#1}}

\newcommand{\acm@contactaddress}{}
\newcommand{\contactaddress}[1]{\renewcommand{\acm@contactaddress}{#1}}

\newcommand{\acm@contactphone}{}
\newcommand{\contactphone}[1]{\renewcommand{\acm@contactphone}{#1}}

\newcommand{\acm@contactfax}{}
\newcommand{\contactfax}[1]{\renewcommand{\acm@contactfax}{#1}}

\newcommand{\acm@contactemail}{}
\newcommand{\contactemail}[1]{\renewcommand{\acm@contactemail}{#1}}

\newcommand{\acm@keywords}{}
\newcommand{\keywords}[1]{\renewcommand{\acm@keywords}{#1}}

\newcommand{\acm@estpages}{}
\newcommand{\estpages}[1]{\renewcommand{\acm@estpages}{#1}}

\newif\ifacm@cover
\acm@coverfalse
%\ifacm@review
%    \acm@covertrue
%\else
%    \acm@coverfalse
%\fi

\newcommand{\suppresscover}{\acm@coverfalse}
\newcommand{\acm@coverpage}{%
    \begin{titlepage}%
        \def\thanks##1{}
        \let\and\acm@origand
        \acm@setninepoint\normalsize
        \renewcommand{\baselinestretch}{1}\selectfont
        \begin{center}%
            \vspace*{\fill}
            {\LARGE\sffamily\bfseries\acm@sectionfont \acm@title \par}%
            \vspace{2\baselineskip}%
            {\large
                \begin{tabular}[t]{c}%
                \acm@author
             \end{tabular}\par%
            }%
            \vspace{1\baselineskip}%
            {\large \acm@affiliation \par}%
            \addvspace{3\baselineskip}%
            {Category: \acm@category \par}%
            \vspace{0.5\baselineskip}%
            {Format: \acm@format \par}%
            \vspace{3\baselineskip}%
            \begin{tabular}{ll}
                Contact: & \acm@contactname \\[1\baselineskip]
                         & \begin{tabular}[b]{@{}l@{}}
                               \acm@contactaddress
                               \end{tabular} \\[1\baselineskip]
                phone: & \acm@contactphone \\
                fax: & \acm@contactfax \\
                email: & \acm@contactemail
            \end{tabular}\par%
            \vspace*{3\baselineskip}%
            {Estimated \# of pages: \acm@estpages \par}%
            \vspace*{\baselineskip}%
            {Keywords: \acm@keywords \par}%
            \vspace*{\fill}%
            \begin{minipage}{5in}%
                \the\acm@abs
            \end{minipage}\par%
            \vspace*{\fill}
        \end{center}%
    \end{titlepage}%
}

\AtEndDocument{\ifacm@cover \acm@coverpage \fi}
\newcommand{\acmopening}[1]{%
  \ClassError{tcvg}{%
    The acmopening command is no longer needed.\MessageBreak%
    Switch to the maketitle command and abstract environment}{}}
\InputIfFileExists{tcvg.cfg}
        {\typeout{***************************************^^J%
                  * Local config file tcvg.cfg used *^^J%
                  ***************************************}}
        {}
\endinput

% End of file 'tcvg.cls'.

