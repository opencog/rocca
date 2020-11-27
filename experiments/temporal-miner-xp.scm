;; Experiment to see if the pattern miner can discover temporal
;; patterns.

(use-modules (opencog))
(use-modules (opencog exec))
(use-modules (opencog spacetime))
(use-modules (opencog ure))
(use-modules (opencog pln))
(use-modules (opencog miner))

;; Parameters

(define minsup 4)				; Minimum support
(define maxiter 10000)				; Maximum number of iterations
(define cnjexp #f)				; Enable conjunction expansion
(define enfspe #t)				; Enable enforce specialization
(define mspc 2)					; Max cnjs to apply specialization
(define maxvars 5)				; Max number of variables
(define surprise 'nisurp)			; Surprisingness measure

;; Loggers
(use-modules (opencog logger))
(cog-logger-set-level! "debug")
(ure-logger-set-level! "debug")
(miner-logger-set-level! "fine")

;; KBs

(define db
  (list
   (AtTime (stv 1 1) (Evaluation (Predicate "P") (Concept "A")) (Z))
   (AtTime (stv 1 1) (Evaluation (Predicate "Q") (Concept "A")) (S (Z)))
   (AtTime (stv 1 1) (Evaluation (Predicate "P") (Concept "B")) (Z))
   (AtTime (stv 1 1) (Evaluation (Predicate "Q") (Concept "B")) (S (Z)))
   (AtTime (stv 1 1) (Evaluation (Predicate "P") (Concept "C")) (S (Z)))
   (AtTime (stv 1 1) (Evaluation (Predicate "Q") (Concept "C")) (S (S (Z))))
   (AtTime (stv 1 1) (Evaluation (Predicate "P") (Concept "D")) (S (Z)))
   (AtTime (stv 1 1) (Evaluation (Predicate "Q") (Concept "D")) (S (S (Z))))
   ))

;; Initial pattern

(define initpat
  (Lambda
    (VariableSet (Variable "$T") (Variable "$X") (Variable "$Y"))
    (Present
      (AtTime (Variable "$X") (Variable "$T"))
      (AtTime (Variable "$Y") (S (Variable "$T"))))))

;; Launch miner

(define results (cog-mine db
                          #:minimum-support minsup
			  #:initial-pattern initpat
                          #:maximum-iterations maxiter
                          #:conjunction-expansion cnjexp
                          #:maximum-variables maxvars
                          #:maximum-spcial-conjuncts mspc
                          #:surprisingness surprise))

;; The expected results should contain
;;
;; (EvaluationLink (stv 0.9375 1)
;;   (PredicateNode "nisurp")
;;   (ListLink
;;     (LambdaLink
;;       (VariableSet
;;         (VariableNode "$T")
;;         (VariableNode "$PM-7731920"))
;;       (PresentLink
;;         (AtTimeLink
;;           (EvaluationLink
;;             (PredicateNode "P")
;;             (VariableNode "$PM-7731920"))
;;           (VariableNode "$T"))
;;         (AtTimeLink
;;           (EvaluationLink
;;             (PredicateNode "Q")
;;             (VariableNode "$PM-7731920"))
;;           (SLink
;;             (VariableNode "$T")))))
;;
;; which translates into
;;
;; (Lambda
;;   (Variable "$X")
;;   (SequentialAnd
;;     (Evaluation (Predicate "P") (Variable "$X"))
;;     (Evaluation (Predicate "Q") (Variable "$X"))))
