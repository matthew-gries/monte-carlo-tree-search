import time
import multiprocessing
class MonteCarloTreeSearch(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        total_simulation_seconds : float
            Amount of time the algorithm has to run. Specified in seconds

        Returns
        -------

        """

        def simulate():  
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        if simulations_number is None :
            assert(total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while time.time() < end_time:
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
        else :
            # make this async
            # for _ in range(0, simulations_number):            
            #     v = self._tree_policy()
            #     reward = v.rollout()
            #     v.backpropagate(reward)
            pool = multiprocessing.Pool(4)

            running = [pool.apply_async(simulate) for _ in range(simulations_number)]
            _ = [f.get() for f in running]

        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):
        """
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
