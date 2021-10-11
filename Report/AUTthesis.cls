
%%
%%  This is file `AUT_thesis.cls',
%%  The original file was xepersian-thesis.cls by Vafa Khalighi
%%   __________________________________
%%   Copyright © 2015-2018 Amirkabir University
%%  
%%   http://www.aut.ac.ir
%%   http://www.parsilatex.com  
%%   http://forum.parsilatex.com
%% 
%% 
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{AUTthesis}
              [08/04/2012 v0.4
 Persian thesis document class in XeLaTeX for University of Amirkabir projects/theses/dissertations]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions
\LoadClass[a4paper,10pt]{book}
\def\department#1{\gdef\@department{#1}}
\def\degree#1{\gdef\@degree{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\thesisdate#1{\gdef\datethesis{#1}}
\newcommand{\firstsupervisor}[1]{\def\ffirstsupervisor{#1}}
\newcommand{\secondsupervisor}[1]{\def\fsecondsupervisor{#1}}
\newcommand{\firstadvisor}[1]{\def\ffirstadvisor{#1}}
\newcommand{\secondadvisor}[1]{\def\fsecondadvisor{#1}}
\newcommand{\internalexaminer}[1]{\def\finternalexaminer{#1}}
\newcommand{\externalexaminer}[1]{\def\fexternalexaminer{#1}}
\newcommand{\fnote}[1]{\LTRfootnote{\lr{#1}}}
\def\name#1{\gdef\fname{#1}}
\def\surname#1{\gdef\lname{#1}}
\def\fatitle#1{\gdef\ftitle{#1}}
\def\keywords#1{\gdef\fkeywords{\noindent #1}}
\def\fa-abstract#1{\gdef\ffa-abstract{#1}}
\def\faculty#1{\gdef\@faculty{#1}}
\def\subject#1{\gdef\fsubject{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\latintitle#1{\gdef\@latintitle{#1}}
\def\latinauthor#1{\gdef\@latinauthor{#1}}
\def\latindegree#1{\gdef\@latindegree{#1}}
\def\latinfaculty#1{\gdef\@latinfaculty{#1}}
\def\latinthesisdate#1{\gdef\@latinthesisdate{#1}}
\def\latinsubject#1{\gdef\@latinsubject{#1}}
%\def\latinfield#1{\gdef\@latinfield{#1}}
\def\firstlatinsupervisor#1{\gdef\@firstlatinsupervisor{#1}}
\def\secondlatinsupervisor#1{\gdef\@secondlatinsupervisor{#1}}
\def\firstlatinadvisor#1{\gdef\@firstlatinadvisor{#1}}
\def\secondlatinadvisor#1{\gdef\@secondlatinadvisor{#1}}
\def\latinname#1{\gdef\@latinname{#1}}
\def\latinsurname#1{\gdef\@latinsurname{#1}}
\def\en-abstract#1{\gdef\een-abstract{#1}}
\def\latinkeywords#1{\gdef\elatinkeywords{#1}}
\def\latinuniversity#1{\gdef\@latinuniversity{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\if@bscthesis
\@bscthesisfalse
\DeclareOption{bsc}{\@bscthesistrue}
\newif\if@mscthesis
\@mscthesisfalse
\DeclareOption{msc}{\@mscthesistrue}
\ProcessOptions
%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\AUTtitle}{\begin{titlepage}
%\topmargin=-30pt
\vspace{-1cm}\centerline{{\includegraphics[height=3.8cm]{figures/fa-logo.png}}}
{\bf{\fontsize{16pt}{16pt}\selectfont دانشگاه صنعتی امیرکبیر}\par
\vspace{-1cm}\fontsize{14pt}{15pt}\selectfont(پلی‌تکنیک تهران)\par
\vspace{-1cm}\fontsize{16pt}{17pt}\selectfont\@faculty}\par
%\vskip 1.5cm
{\fontsize{16pt}{14pt}\selectfont\bfseries
\if@mscthesis
پایان‌نامه کارشناسی‌ارشد
\vspace{-.8cm}

\@department
\else
\if@bscthesis
پروژه کارشناسی
\vspace{-.8cm}

\@department
\else
رساله دکتری
\vspace{-.8cm}

\@department
\fi\fi
}

% {\fontsize{16pt}{17pt}\selectfont\bfseries%
%  \onvan}\par
\par
 \vspace{1cm}
{\fontsize{20pt}{21pt}\selectfont\bfseries {\ftitle}}\par
\vskip .5cm
 {\fontsize{16pt}{17pt}\selectfont\bfseries%
  \by}\par
\par
\vspace*{-0.8cm}

\fontsize{16pt}{17pt}\selectfont\bfseries
% نگارش: 
% \\[0.8cm]
% \vspace*{-0.8cm}
{\fontsize{18pt}{19pt}\selectfont\bfseries{\fname \lname}}\par
% \vspace*{-0.8cm}

% {\fontsize{18pt}{19pt}\selectfont\bfseries{\fname \lname}}\par
\fontsize{16pt}{17pt}\selectfont\bfseries
{\ifx\ffirstsupervisor\undefined%
\ifx\fsecondsupervisor\undefined%
\else\fi
\else
\ifx\fsecondsupervisor\undefined%
استاد راهنما:
\vspace*{-0.8cm}
{\fontsize{18pt}{19pt}\selectfont\bfseries\par
\ffirstsupervisor}
\else
\fontsize{16pt}{17pt}\selectfont\bfseries
استادان راهنما
\vspace*{-0.8cm}
{\fontsize{18pt}{19pt}\selectfont\bfseries\par \ffirstsupervisor\space
	 و
	  \fsecondsupervisor}
\fi\fi}
\par\fontsize{16pt}{17pt}\selectfont\bfseries
{\ifx\ffirstadvisor\undefined%
\ifx\fsecondadvisor\undefined%
\else\fi
\else
\ifx\fsecondadvisor\undefined%
استاد مشاور
\vspace*{-0.8cm}
{\fontsize{18pt}{19pt}\selectfont\bfseries\par
\ffirstadvisor}
\else

\fontsize{16pt}{17pt}\selectfont\bfseries
استادان مشاور
\vspace*{-0.8cm}
{\fontsize{18pt}{19pt}\selectfont\bfseries\par\ffirstadvisor{} و \fsecondadvisor}
\fi\fi}

\vskip 6mm

{\fontsize{16pt}{17pt}\selectfont \datethesis}

\end{titlepage}%
\newpage
\thispagestyle{empty}
\baselineskip=.7cm
\if@bscthesis \relax
\else
%-----------------------------------------------------------------------------------table of thesis
\relax
\fi
}
%دستوری برای لیست نمادها 
\newcommand{\symb}[2]{\makebox[10cm]
{\makebox[1cm][c]{\hfill}\makebox[3cm][r]{$\displaystyle #1$}\makebox[6cm][r]{#2}}\newline}

\newcommand{\exam}[2]{\makebox[12cm]
{\makebox[2cm][c]{\hfill}\makebox[4cm][r]{#1} \ \  \makebox[6cm][r]{#2}}\newline}
% دستوری برای تعریف واژه‌نامه انگلیسی به فارسی
\newcommand{\englishTOfarsi}[2]{\par\bigskip\rl{#2}\dotfill\lr{#1}\par} 
% دستوری برای تعریف واژه‌نامه فارسی به انگلیسی 
\newcommand{\farsiTOenglish}[2]{\par\bigskip#1\dotfill\lr{#2}\par}
%-----------------------------------------------
\def\frame{\newpage
\thispagestyle{empty}
\begin{framed}\thispagestyle{empty}
\def\endframe{\thispagestyle{empty}}\end{framed}
\thispagestyle{empty}
\newpage
}
\def\abstractpage{\newpage
\thispagestyle{empty}
\vskip 15mm
\par
\begin{abstract}}
\def\endabstractpage{\end{abstract}
\newpage
}

\def\latinabstract{\newpage
\thispagestyle{empty}
\vskip 15mm
\begin{center}{\fontsize{16pt}{17pt}\selectfont\bfseries \@latintitle \\[5mm]}
{\bfseries Abstract}
\end{center}
\vspace{5mm}}

\def\endlatinabstract{\newpage}

\def\acknowledgementpage{\newpage
\thispagestyle{empty}

\vspace{1cm}
\par\noindent}
\def\endacknowledgementpage{\newpage}

\def\titlepage{\newpage\centering
  \thispagestyle{empty}
  \parindent 0pt \parskip 10pt plus 1fil minus 1fil
  \def\baselinestretch{1}\@normalsize\vbox to \vsize\bgroup\vbox to 9in\bgroup}
\def\endtitlepage{\par\kern 0pt\egroup\vss\egroup\newpage}

\def\signature{\vspace{1cm}
\begin{flushleft}
{\fname\lname \\
\datethesis}
\end{flushleft}}

\def\fullname
{\fname \lname }

\def\abstract{\subsection*{\abstractname}\small
\@normalsize}
\def\endabstract{\par}

\font\titlefont=cmssbx10 scaled 2074
\font\supervisorfont=cmbxti10

\newcommand{\latinvtitle}{%
\if@bscthesis \relax
\else
\relax
%-------------------------------------------------------------------------- table of En thesis

%---------------------------------------------------------------------------- end of En table
\begin{titlepage}
\vspace{-1cm}{\includegraphics[height=3.8cm]{en-logo}}
\centerline{\fontsize{18pt}{19pt}\selectfont\bf Amirkabir University of Technology}\space 
\vskip -.2cm
\fontsize{16pt}{17pt}\selectfont\bfseries (Tehran Polytechnic)\par \vskip .2cm
\bfseries
\fontsize{16pt}{17pt}\selectfont\@latinfaculty\space 
\vskip 5mm
 \if@mscthesis
M. Sc. Thesis\par
\else
Ph. D Thesis\par
\fi\par
\vspace*{2cm}
%{\fontsize{16pt}{17pt}\selectfont\bfseries Title}%\@latinsubject
\par
\baselineskip=.750cm
{\Huge\@latintitle}%\titlefont 
\baselineskip=.750cm \par
\vskip 1.5cm
{\bfseries\latinby}\par
{\fontsize{18pt}{19pt}\selectfont\bfseries
\@latinname \; \@latinsurname}\par
\vskip 1cm

\fontsize{16pt}{17pt}\selectfont\bfseries
{\ifx\@firstlatinsupervisor\undefined%
\ifx\@secondlatinsupervisor\undefined%
\else\fi
\else
\ifx\@secondlatinsupervisor\undefined%
Supervisor
{\fontsize{18pt}{19pt}\selectfont\bfseries\par
\@firstlatinsupervisor}
\else
\fontsize{16pt}{17pt}\selectfont\bfseries
Supervisors
\par
{\fontsize{18pt}{19pt}\selectfont\bfseries \@firstlatinsupervisor{} and \@secondlatinsupervisor}
\fi\fi}
\par
\vskip 1cm
\fontsize{16pt}{17pt}\selectfont\bfseries
{\ifx\@firstlatinadvisor\undefined%
\ifx\@secondlatinadvisor\undefined%
\else\fi
\else
\ifx\@secondlatinadvisor\undefined%
Advisor
{\fontsize{18pt}{19pt}\selectfont\bfseries\par
\@firstlatinadvisor}
\else

\fontsize{16pt}{17pt}\selectfont\bfseries
Advisors
\par
{\fontsize{18pt}{19pt}\selectfont\bfseries\@firstlatinadvisor{} and \@secondlatinadvisor}
\fi\fi}
%\vskip 5mm

\par
\vskip 2cm

{\fontsize{16pt}{17pt}\selectfont\bfseries\@latinthesisdate}
\par
\vskip 1cm

\par

\vfill
%
\end{titlepage}
\fi}

\def\by{نگارش:}
\def\latinby{By}
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
\hbox{}
\thispagestyle{empty}
\newpage
\if@twocolumn\hbox{}\newpage\fi\fi\fi}

%% 
%% Copyright © 2015-2018   
%% 
%% Distributable under the LaTeX Project Public License,
%% version 1.3c or higher (your choice). The latest version of
%% this license is at: http://www.latex-project.org/lppl.txt
%% 
%% This work is "maintained" (as per LPPL maintenance status)
%% 
%% 
%% 
%% 
%% 
%% Special thanks to Vafa Khalighi for his invaluable hints.
%%
% End of file `AUT_thesis.cls'.
