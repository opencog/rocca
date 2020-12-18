# Proto-AGI early progress report and planning (2020-12-17)

Here's a sum-up (not an indepth report, that will come later) of
what's been done so far with opencog-gym, its current limits and what
remains to be done.  It is writen from the perspective of Legacy
OpenCog, even though likely a good portion of it applies to OpenCog
Hyperon as well.

opencog-gym is 

## What do we have so far?

We have an OpenCog agent [1], easily wrappable to any synchronous
environment with a gym-like API, that can:

1. Learn (via the pattern miner) cognitive schematics (predictive
   implications) with arbitrary pre/post contexts, and one or more
   actions.

2. Use these cognitive schematics as plans to make decisions in a
   trivial environment (chase a food pellet in a 2-square board).

## What are the limits?

1. Consecutive lags in predictive implications and sequential ands are
   limited to one unit of time, i.e. can capture cognitive schematics
   such as

   """ if context C is true at T, and action A1 is executed at T, and
   if action A2 is executed at T+1, then goal G is likely to be
   acheived at T+2. """

   Arbitrary lags, for mining or else, are not supported.

2. Mining is fairely inefficient due to having to use Naturals to be
   able to match lags between events.  For instance 3 units of time is
   represented as

   (SLink (SLink (SLink (ZLink))))

   Or a lag of 2 units of time relative to (Variable "$T") is
   represented as

   (SLink (Slink (VariableNode "$T")))

3. Another source of inefficiency is decision making.  Although it is
   considerably more efficient than learning/planning (for now done
   with the miner, then soon done with PLN as well).  So once plans
   are there, decision is relatively fast, though in the current
   setting still takes 3 seconds to select an action given cognitive
   schematics to select from.  In comparision discovering these 200
   cognitive schematics takes over a minute.

4. And of course the whole thing is inefficient due to the lack of
   1. SpaceTime server or alike;
   2. ECAN (Attention Allocation);
   3. Ultimately, capacity to create/use abstractions for modeling and
      acting. Schematization/specialization, meta-learning, etc.

5. As of right now a goal cannot be decomposed into subgoals, this
   requires futher temporal logic development, currently in progress.

## What remains to be done?

1. Move from the simplistic Chase environment to Malmo.  According to
   Kasim the plumbing is ready.  Maybe for starter we could port the
   Chase env to Malmo.  Then, as Alexey suggested [2] we need to
   create high level actions, such as goto(X), etc, as well as high
   level perceta.

2. Introduce temporal deduction.  The agent should be able to chain
   actions sequences that it has never observed, but rather based on
   whether the post-context of some cognitive schematics matches with
   (implies or inherits from) the pre-context of another.  For that we
   need to implement a similar deduction rule as [3] for predictive
   implications.  This is also required for decomposing goals into
   subgoals.

3. Extend the support to arbitrary time lags.  The agent should be
   able to notice surprising combinations of events even if they are
   arbitrary lagging. Of course some heuristics could be used to
   lesser their prior as the lag increases, or if it is too distant
   from the "scale" of the events (a notion that remains to be defined
   but should be doable).  For that to happen it looks like we need
   two things:

   1. The pattern miner should handle time intervals, then the same a
      priori property used to prune the search tree [2] could be used
      with time intervals, the larger the time interval the more
      abstract the pattern.

   2. We likely need to represent the distribution of lag (based on
      evidence or indirectly inferred).  One possibility would be to
      introduce a temporal truth value (as suggested in [3]), using a
      distributional lag.  As of today the lag is represented in the
      temporal relationship instead, see [4], which makes sense given
      the current limits as it allows to represent the same predictive
      implication with different lags, resulting in different truth
      values, but once a distributional lag is used there is no longer
      the need for it.  Even the TV of an "eventual" predictive
      implication can be obtained by looking at the extremum the
      cumulative distribution function.  So that way only one
      predictive implication solely determined by its pre and post
      contexts is required.  Of course such distributional temporal
      truth value could inherit from a more general distribution truth
      value.

4. Improve mining efficiency.  We could (with Legacy OpenCog in mind):

   1. Upgrade the pattern matcher to have queries like

      (Get (AtTime A (Plus (TimeNode "1" (TimeNode "10")))))

      be able to match

      (AtTime A (TimeNode "11"))

   In that case timestamps would still be in the atomspace, which is
   inheritantly inefficient, so maybe we should go with:

   2. Support the spacetime server or such.  Maybe the one in the
      spacetime repository, possibly combined with atom values.  This
      would be more efficient and enable more flexible matching
      involving time intervals.  It would however require to either
      fiddle with the pattern matcher or the pattern miner to
      integrate that type of temporal queries.

   Of course similar tasks could take place on OpenCog Hyperon but the
   specifics remain to be defined.

   Beside these "low level" improvements there are certainly a number
   of algorithmic improvements that can be done.  I have compiled a
   number of papers (that I haven't read) on the subject, might or not
   be relevant:

   - https://cs.fit.edu/~pkc/classes/ml-internet/papers/cao05icdm.pdf
   - http://fuzzy.cs.ovgu.de/wiki/uploads/Mitarbeiter.Kempe/FSMTree.pdf
   - https://www2.cs.sfu.ca/~jpei/publications/Prob%20Spatio-temp%20patterns%20ICDM13.pdf
   - https://people.cs.pitt.edu/~milos/research/KDD_2012_recent_patterns.pdf
   - https://arxiv.org/pdf/1804.10025v1.pdf
   - http://users.ics.aalto.fi/panagpap/KAIS.pdf
   - http://www.jatit.org/volumes/Vol68No2/4Vol68No2.pdf
   - https://eventevent.github.io/papers/EVENT_2016_paper_16.pdf
   - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4192602/
   - https://arxiv.org/pdf/2010.03653.pdf

5. Improve decision efficiency.  The decision procedure (consisting on
   estimating the posterior of each cognitive schematic, followed by
   Thomposon Sampling) is entirely implemented in Python, which partly
   explains its slowness (3 seconds for 200 cognitive schematics).
   Re-implementing in C++ or Rust would certainly speed it up, however
   before that we want to optimize the algorithm itself, maybe
   introduce ECAN, maybe filter by surprisingness and prior, etc.

6. Add ECAN support.  Being unfamiliar with ECAN's implementation I do
   not know the specifics of that tasks, beside of course the first
   step of getting familiar with it.

7. Regarding concept creation, meta-learning, etc, we can probably
   worry about it when most of the improvements above have been done.
   The idea yet is to record internal cognitive operations taking
   place during learning, planning, etc, find patterns and plan
   internal actions (tweak this parameter, rewrite this rule, etc)
   that provably increase the likelihood of goal fulfillment.  This
   aspect is also important in order to take into account the temporal
   contrains of the planning process. I.e. if a goal must be acheived
   within 10s, then a crude meta-reasoning could take place to
   evaluate that reasoning for say 2s, then executing the resulting
   plan for the remaining 8s is likely to succeed.  By considering
   "planning" and "executing plan" as high level actions, this should
   be possible without much radical change.

8. I have not mentioned reasoning and inference control because as of
   now PLN is only used very shallowly (to calculate TVs of predictive
   implications based on evidence).  More attention will be required
   once futher development has taken place on that front.

9. Last but not least we need unit and integration tests.

## References

- [1] https://github.com/singnet/opencog-gym
- [2] https://docs.google.com/document/d/1gvYo8zo501Tu6o4xJbx1OLh3gtRldSPNiRxHD8ZTcLA/edit
- [3] https://github.com/singnet/pln/blob/master/opencog/pln/rules/term/deduction.scm
- [4] Fast Discovery of Association Rules, R. Agrawal et al.
- [5] Chapter 14 of the PLN book http://goertzel.org/PLN_BOOK_6_27_08.pdf
- [6] https://wiki.opencog.org/w/PredictiveImplicationLink
