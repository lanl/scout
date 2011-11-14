;;
;; -----  Scout Programming Language -----
;;
;; This file is distributed under an open source license by Los Alamos
;; National Security, LCC.  See the file License.txt (located in the
;; top level of the source distribution) for details.
;; 
;; -----
;; 
;; Major mode for editing Scout programs in emacs. 
;; 

(require 'cc-mode)
(eval-when-compile 
  (require 'cc-langs)
  (require 'cc-fonts))


(eval-and-compile 
  ;; make sure scout-mode is known to the language constant system.
  ;; We use C++ mode as the fallback for constants we don't change 
  ;; in scout-mode.  This also needs to be done at compile time...
  (c-add-language 'scout-mode 'c++-mode))

(c-lang-defconst c-primitive-type-kwds
  "Primitive type keywords.  As oposed to other keyword lists,
the keywords listed here are fontified with the type face vs.
the keyword face.

If any of these are also on the `c-type-list-kwds',
`c-ref-list-kwds', `c-colon-type-list-kwds',
`c-paren-nontype-keywds', `c-paren-type-kwds', `c-<>-type-kwds',
or `c-<>-arglist-kwds' then the associated clauses will be
handled.

Do not try to modify this list for end-user customizations; the 
`*-font-lock-extra-types' variable, where `*' is the mode prefix,
is the appropriate place for that."
  scout
  (append 
   '("char4"    "char3"    "char2"
     "short4"   "short3"   "short2"
     "int4"     "int3"     "int2"
     "long4"    "long3"    "long2"
     "float4"   "float3"   "float2"
     "double4"  "double3"  "double2")
   ;; careful to not be destructive here -- use append...
   (append 
    (c-lang-const c-primitive-type-kwds)
    nil)))



(c-lang-defconst c-protection-kwds
  "Scout topological keywords in meshes."
  t nil 
  scout  (append '("cells" "vertices" "edges" "faces")
		 (append  (c-lang-const c-protection-kwds)
			  nil)))

(c-lang-defconst c-type-prefix-kwds
  t    nil
  scout  (append '("mesh")
		 (c-lang-const c-type-prefix-kwds c++)))

(c-lang-defconst c-block-decls-with-vars 
  t nil
  scout (append '("mesh")
		(c-lang-const c-block-decls-with-vars c++)))

(c-lang-defconst c-block-stmt-2-kwds
  "keywords followed by substatements."
  scout (append '("foreach" "forall" "renderall")))

(c-lang-defconst c-type-modifier-kwds
  "Mesh type modifier keywords."
  scout  (append '("uniform" "rectilinear" "structured" "unstructured")
   (c-lang-const c-modifier-kwds)
   nil))

(c-lang-defconst c-primary-expr-kwds
  "Keywords besides constants and operators that start primary expressions."
  scout  '("cells" "vertices" "edges" "faces"))

(c-lang-defconst c-paren-nontype-kwds
  "Keywords that may be followed by a parenthetical expression that
does not contain type identifiers."
  scout (append '("of" "in")))


(c-lang-defconst c-class-decl-kwds 
  "class declaration related keywords"
  scout (append '("mesh")))

(defcustom scout-font-lock-etra-types nil
  "*List of extra types (aside from keywords) to recognize in scout-mode.
Each list item should be a regexp matching a single identifier.")


(defconst scout-font-lock-keywords-1
  (c-lang-const c-matchers-1 scout)
  "Minimal highlighting for scout-mode.")

(defconst scout-font-lock-keywords-2
  (c-lang-const c-matchers-2 scout)
  "Fast normal highlighting for scout-mode.")

(defconst scout-font-lock-keywords-3
  (c-lang-const c-matchers-3 scout)
  "Accurate normal highlighting for scout-mode.")

(defvar scout-font-lock-keywords scout-font-lock-keywords-3
  "Default expressions to highlight in scout-mode.")


(defvar scout-mode-syntax-table nil
  "Syntax table used in scout-mode buffers.")
(or scout-mode-syntax-table
    (setq scout-mode-syntax-table
    (funcall (c-lang-const c-make-mode-syntax-table scout))))

(defvar scout-mode-abbrev-table nil
  "Abbreviation table used in scout-mode buffers.")

(c-define-abbrev-table 'scout-mode-abbrev-table 
  ;; keywords that if they occur first on a line might
  ;; alter the syntatic context, and thus should trigger 
  ;; indentation when completed...
  '(("else" "else" c-electric-continued-statement 0)
    ("while" "while" c-electric-continued-statement 0)))

(defvar scout-mode-map (let ((map (c-make-inherited-keymap)))
			 ;; Add bindings that are only useful for Scout.
			 map)
  "Keymap used in scout-mode buffers.")


(easy-menu-define scout-menu scout-mode-map "Scout Mode Commands"
  ;; Can use 'scout' as the language for `c-mode-menu' because the
  ;; definition covers any language.  In this case, the language is
  ;; used to adapt to the nonexistence of a c++ pass and thus removing
  ;; some irrelevant menu alternatives. 
  (cons "Scout" (c-lang-const c-mode-menu scout)))


;;;###autoload
(add-to-list 'auto-mode-alist '("\\.sch\\'" . scout-mode))
(add-to-list 'auto-mode-alist '("\\.sc\\'"  . scout-mode))

;;;###autoload
(defun scout-mode ()
  "Major mode for editing Scout -- a domain-specific embedding
  within C/C++.  The hook `c-mode-common-hook' is run with no
  arguments at mode initialization, followed by
  `scout-mode-hook'.

Key bindings:
\\{scout-mode-map}"
  (interactive)
  (kill-all-local-variables)
  (c-initialize-cc-mode t)
  (set-syntax-table scout-mode-syntax-table)
  (setq major-mode 'scout-mode
	mode-name "scout"
	local-abbrev-table scout-mode-abbrev-table
	abbrev-mode t)
  (use-local-map c-mode-map)
  ;; `c-init-language-vars' is a macro that is expanded at compile
  ;; time to a large `setq' with all the language variables and their
  ;; customized values for scout.
  (c-init-language-vars scout-mode)
  ;; `c-common-init' initializes most of the components of a CC Mode
  ;; buffer, including setup of the mode menu, font-lock, etc. There
  ;; is also a lower level routine `c-basic-common-init' that only
  ;; initializes the syntactic analysis and supporting portions...
  (c-common-init `scout-mode)
  (easy-menu-add scout-menu)
  (run-hooks 'c-mode-common-hook)
  (run-hooks 'scout-mode-hook)
  (setq font-lock-keywords-case-fold-search t)
  (c-update-modeline))

(provide 'scout-mode)

