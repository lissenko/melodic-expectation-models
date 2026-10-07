;;;; Custom viewpoints for IDyOM. Load after IDyOM: (load "idyom/viewpoints.lisp").
;;;;
;;;; bioi-ratio-q: IDyOM's bioi-ratio rounded to the nearest rhythm category in
;;;; log2 space, codes 0-12 (0 below 1/4, 12 above 4). Identical boundaries to
;;;; the networks' bioi_ratio feature (models/features.py, IOI_RATIO_BOUNDS).
;;;; Needed because the corpus is millisecond-timed: its raw bioi-ratio has
;;;; thousands of distinct values, while the stimuli's are exact simple ratios.
;;;; Boundaries are irrational, so no ratio of integers falls on one and float
;;;; rounding cannot split the two implementations.

(cl:in-package #:viewpoints)

(defparameter *bioi-ratio-q-bounds*
  (let* ((centres (mapcar (lambda (r) (log (float r 1d0) 2d0))
                          '(1/4 1/3 1/2 2/3 3/4 1 4/3 3/2 2 3 4)))
         (mids (mapcar (lambda (a b) (/ (+ a b) 2d0)) centres (cdr centres))))
    (append (list (- (first centres) (/ (- (second centres) (first centres)) 2d0)))
            mids
            (let ((l (car (last centres))) (p (car (last centres 2))))
              (list (+ l (/ (- l p) 2d0)))))))

(defun bioi-ratio-q-code (ratio)
  (let ((x (log (float ratio 1d0) 2d0)))
    (count-if (lambda (b) (<= b x)) *bioi-ratio-q-bounds*)))

(define-viewpoint (bioi-ratio-q derived (bioi))
    ((events md:melodic-sequence) element)
  :function (let ((r (bioi-ratio events)))
              (if (or (undefined-p r) (not (plusp r))) +undefined+
                  (bioi-ratio-q-code r)))
  :function* (let ((prev (bioi (list (penultimate-element events)))))
               (remove-if-not #'(lambda (a) (and (plusp prev) (plusp a)
                                                 (= (bioi-ratio-q-code (/ a prev)) element)))
                              (viewpoint-alphabet (get-viewpoint 'bioi)))))

;;;; bioi-contour-q: shorter / same / longer (-1 0 1) than the previous
;;;; inter-onset interval, read off bioi-ratio-q so that "same" means the "1"
;;;; category (code 6) rather than exact equality, which the millisecond-timed
;;;; corpus rarely produces. Matches the networks' bioi_contour feature.

(defparameter *bioi-ratio-q-one* 6)

(define-viewpoint (bioi-contour-q derived (bioi))
    ((events md:melodic-sequence) element)
  :function (let ((c (bioi-ratio-q events)))
              (if (undefined-p c) +undefined+
                  (signum (- c *bioi-ratio-q-one*))))
  :function* (let ((prev (bioi (list (penultimate-element events)))))
               (remove-if-not #'(lambda (a) (and (plusp prev) (plusp a)
                                                 (= (signum (- (bioi-ratio-q-code (/ a prev))
                                                               *bioi-ratio-q-one*))
                                                    element)))
                              (viewpoint-alphabet (get-viewpoint 'bioi)))))

;;;; Alphabets for chunked prediction. IDyOM builds the alphabet of every basic
;;;; viewpoint from the training set plus the dataset being predicted
;;;; (resampling:idyom-resample -> get-basic-viewpoints). When a corpus slice is
;;;; predicted in chunks (training/make_idyom_chunks.py), *alphabet-extra-ids*
;;;; names the whole slice, so each chunk gets exactly the alphabet a single run
;;;; over the slice would have, and chunked output equals the single run.

(defvar *alphabet-extra-ids* nil)

(defun get-basic-viewpoints (attributes dataset)
  (initialise-basic-viewpoints
   (if *alphabet-extra-ids*
       (append dataset (md:get-music-objects *alphabet-extra-ids* nil))
       dataset))
  (get-viewpoints attributes))
