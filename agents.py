import random
import math


BOT_NAME = "himbo, working title"


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
    def __init__(self, sd=None):
        if sd is None:
            self.st = None
        else:
            random.seed(sd)
            self.st = random.getstate()

    def get_move(self, state):
        if self.st is not None:
            random.setstate(self.st)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move."""
    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    # TODO
    def minimax(self, state):
        """Determine the minimax utility value of the given state. Should recursively traverse the game
        tree, eventually determining the value of  that stae for use in the get_move() method. 

        Traverse to leaf state, endgame condition. Alternate between max and min. 
        Args:
            state: a connect383.GameState object representing the current board

        Returns: the exact minimax utility value of the state
        """
        # base case: no more possible moves, return value up tree
        if (state.is_full()):
            return state.utility()
        values = []
        successors = state.successors()
        # Player 1's turn: continue to recurse down and then maximize
        if (state.next_player() == 1):
            for succ in successors:
                # succ[0] is action, succ[1] is state
                next_move = succ[1]
                # recursion go skrrrrra 
                values.append(self.minimax(next_move)) 
            return max(values)
        # Player 2's turn: continue to recurse through and then minimize
        else: 
            for succ in successors:
                next_move = succ[1]
                values.append(self.minimax(next_move)) 
            return min(values)
        print("error!!! double check")
        return 42  # We should never get to this line

# for succ in successors:
#     print (succ[1].__str__())
class MinimaxHeuristicAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    # TODO
    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state.

        The depth data member (set in the constructor) determines the maximum depth of the game 
        tree that gets explored before estimating the state utilities using the evaluation() 
        function.  If depth is 0, no traversal is performed, and minimax returns the results of 
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        #
        # If you get to a leaf node you should always return the actual value. 
        # There's no uncertainty about the value, since it is a leaf. 
        # So, no need to run an evaluation function.
        #
        if (self.depth_limit == None):
            # inherit parent
            return MinimaxAgent.minimax(state)

        return self.minimaxHelper(state, self.depth_limit)

    def minimaxHelper(self, state, cur_depth):
        # Base case: If you get to a leaf node you should always return the actual value. 
        if (state.is_full()):
            # to test: use a high depth on a small board
            return state.utility()
        # Base case: reached depth 0
        if (cur_depth == 0):
            return self.evaluation(state)
        
        # Recursive step
        else:
            values = []
            successors = state.successors()
            # Player 1's turn: continue to recurse down and then maximize
            if (state.next_player() == 1):
                for succ in successors:
                    next_move = succ[1]
                    # recursion hit you with the DDU-DU DDU-DU
                    values.append(self.minimaxHelper(next_move, cur_depth - 1)) 
                return max(values)
            # Player 2's turn: continue to recurse through and then minimize
            else: 
                for succ in successors:
                    next_move = succ[1]
                    values.append(self.minimaxHelper(next_move, cur_depth - 1)) 
                return min(values)
            return 42  # Shouldn't reach this line


    # TODO
    def evaluation2(self, state):
        """Estimate the utility value of the game state based on features.

        N.B.: This method must run in O(1) time!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heusristic estimate of the utility value of the state
        """
        """
        Ideas:
            count open ended 2 in a rows for each player for a given state, and return the difference
            count number of spaces around streaks - stanford pdf online 
            count how many rows i am blocking ( if a piece can block 2 opponent rows of 3, better than block
                single row of 2)
            count how rows i'm filling in (placing a piece that forms 2 rows of 3 is better than a 
                piece that fills one row of 3)
            takes a state in the tree and tells you how good it is 
            figure out some things you care about: 3 in a rows or maye diagonals are better 

            number of diagonal 3 in a rows and it gets a weight of 0.3 * # diag + 0.2 # horizontals
            scores is an evaluation 
            here's what the most promising features of the state are. 
        """
        s1, s2 = state.scores()
        # weigh 3 in a rows more than two in a rows
        
        p1_two = 0
        p2_two = 0
        p1_three_plus = 0
        p2_three_plus = 0
        # we favor creating more 2 in a rows and especially favor 3+ in a row
        for run in state.get_diags():
            for elt, length in diags(run):
                if (elt == 1) and (length >= 3):
                    p1_three_plus += 2
                elif (elt == 1) and (length == 2):
                    p1_two += 1
                elif (elt == -1) and (length >= 3):
                    p2_three_plus += 2
                elif (elt == -1) and (length == 2):
                    p2_two += 1
        return (s1 + p1_three_plus + p1_two) - (s2 + p2_three_plus + p2_two) #  iunno what im doing


    def evaluation(self, state):
        estimated_util = 0

        rows = state.get_rows()
        cols = state.get_cols()
        diag = state.get_diags()

        def calc_util(dimension):
            score = 0
            for dim in dimension:
                grouped = []
                curr = dim[0]
                index = 0
                # slice grouped elements into their own arrays
                for e in range(1, len(dim)):
                    if dim[e] != curr:
                        grouped.append(dim[index : e])
                        index = e
                        curr = dim[e]
                grouped.append(dim[index : len(dim)])

                for i in range(0, len(grouped)):
                    oe = False
                    if (i - 1 > -1) and (grouped[i] == 1 or grouped[i] == -1):
                        if grouped[i - 1][0] == 0:
                            oe = True
                    if (i + 1 < len(grouped)) and (grouped[i] == 1 or grouped[i] == -1):
                        if grouped[i + 1][0] == 0:
                            oe = True
                    if oe and len(grouped) >= 2:
                        score += (len(grouped[i]) + 1) * grouped[i][0]
                    elif not oe and len(grouped) >= 3:
                        score += (len(grouped[i]) - 1) * grouped[i][0]
            return score 

        estimated_util += calc_util(rows)
        estimated_util += calc_util(cols)
        estimated_util += calc_util(diag)

        return estimated_util # Change this line!


def diags(lst):  
    """Get the lengths of all the streaks of the same element in a sequence."""
    rets = []  # list of (element, length) tuples
    prev = lst[0]
    curr_len = 1
    for curr in lst[1:]:
        if curr == prev:
            curr_len += 1
        else:
            rets.append((prev, curr_len))
            prev = curr
            curr_len = 1
    rets.append((prev, curr_len))
    return rets



class MinimaxHeuristicPruneAgent(MinimaxHeuristicAgent):
    """Smarter computer agent that uses minimax with alpha-beta pruning to select the best move."""

    # TODO
    def minimax(self, state):
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the value of the class 
        variable GameState.state_count, which keeps track of how many GameState objects have been 
        created over time.  This agent should also respect the depth limit like HeuristicAgent.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to to column 1.
        
        Args: 
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        # if (state.is_full()):
        #     return state.utility()
        
        # TODO: WHAT HAPPENS IN THIS CASE?
        # if (self.depth_limit == None):
        #     # inherit parent
        #     return MinimaxAgent.minimax(state)

        # call alphabeta_prune helper hehehe
        
        return self.alphabeta_prune(state, self.depth_limit, -math.inf, math.inf)
        
    
    def alphabeta_prune(self, state, cur_depth, low, high):
        # base case: state is full, just take utility.
        if (state.is_full()):
            # print(type(self.depth_limit))
            # print("do i even get here??")
            # to test: use a high depth on a small board
            return state.utility()

        # base case: depth limit reached
        if (cur_depth == 0):   
            return self.evaluation(state)

        # cur_range represents the least and greatest values that the current state can take
        # math.inf are just temporary values, and they will be changed/compared after recursive steps
        cur_low, cur_high = -math.inf, math.inf
        cur_range = [cur_low, cur_high]

        # print("am i even in here?")
        successors = state.successors()
        # Player 1's turn: continue to recurse down and then compare cur min to parent's max
        if (state.next_player() == 1):
            for succ in successors:
                next_move = succ[1]

                # recursion go skrrrrra  
                temp = self.alphabeta_prune(next_move, cur_depth - 1, cur_low, cur_high)

               # if successor's util/eval > than cur range's max, set it as new cur_max
                if (temp > cur_low):
                    cur_low = temp

                # pruning step: cur_low of this state's range is greater than parent's max -> NO overlap: return early
                if (cur_low > high):
                    return cur_low
            # print(cur_low)
            return cur_low

        # Player 2's turn: continue to recurse through and then compare cur max to parent's min
        else: 
            for succ in successors:
                next_move = succ[1]

                # recursive step: look at child until base caseo of depth = 0 or   
                temp = self.alphabeta_prune(next_move, cur_depth - 1, cur_low, cur_high)

               # if successor's util/eval is less than cur range's max, set it as new cur_low
                if (temp < cur_high):
                    cur_high = temp

                # pruning step: cur_high of this state's range is less than parent's low -> NO overlap: return early
                if (cur_high < low):
                    return cur_high
            # print(cur_high)
            return cur_high
        return 42  # Shouldn't reach this line