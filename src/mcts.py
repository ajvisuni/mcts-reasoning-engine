import math
from typing import Dict, List, Optional, Protocol, Tuple


class ModelInterface(Protocol):
    def expand_candidates(self, state: str, k: int = 4) -> List[Tuple[str, str, float, bool]]:
        ...

    def evaluate_prm(self, states: List[str]) -> List[float]:
        ...


class MCTSNode:
    def __init__(
        self,
        state: str,
        parent: Optional["MCTSNode"] = None,
        action: Optional[str] = None,
        prior: float = 1.0,
        is_terminal: bool = False,
        is_solved: bool = False,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.is_terminal = is_terminal
        self.is_solved = is_solved

        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prm_value: float = 0.0
        self.children: Dict[str, "MCTSNode"] = {}
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def puct_score(self, c_puct: float = 1.4) -> float:
        parent_n = self.parent.visit_count if self.parent is not None else 1
        u = c_puct * self.prior * (math.sqrt(max(1, parent_n)) / (1 + self.visit_count))
        return self.q_value + u

    def extract_action_trace(self) -> List[str]:
        trace = []
        curr: Optional[MCTSNode] = self
        while curr is not None and curr.action is not None:
            trace.append(curr.action)
            curr = curr.parent
        return list(reversed(trace))


class MCTSEngine:
    def __init__(self, model: ModelInterface, c_puct: float = 1.4, branch_factor: int = 4):
        self.model = model
        self.c_puct = c_puct
        self.branch_factor = branch_factor

    def select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.is_expanded and len(node.children) > 0 and not node.is_terminal:
            node = max(node.children.values(), key=lambda c: c.puct_score(self.c_puct))
        return node

    def expand_and_evaluate(self, node: MCTSNode) -> Tuple[MCTSNode, float]:
        if node.is_terminal:
            return node, (1.0 if node.is_solved else 0.0)

        candidates = self.model.expand_candidates(node.state, k=self.branch_factor)
        node.is_expanded = True

        if not candidates:
            node.is_terminal = True
            return node, 0.0

        child_states = [c[1] for c in candidates]
        prm_scores = self.model.evaluate_prm(child_states)

        for (action_s, next_s, prior_p, is_solved), prm_v in zip(candidates, prm_scores):
            is_term = is_solved or next_s.strip().endswith("<EOS>")
            val = 1.0 if is_solved else (0.0 if is_term else float(prm_v))
            child = MCTSNode(
                state=next_s,
                parent=node,
                action=action_s,
                prior=prior_p,
                is_terminal=is_term,
                is_solved=is_solved,
            )
            child.prm_value = val
            node.children[action_s] = child
            if is_solved:
                return child, 1.0

        best_child = max(node.children.values(), key=lambda c: c.prm_value + 0.2 * c.prior)
        return best_child, best_child.prm_value

    def backpropagate(self, node: MCTSNode, value: float) -> None:
        curr: Optional[MCTSNode] = node
        while curr is not None:
            curr.visit_count += 1
            curr.value_sum += value
            curr = curr.parent

    def search(
        self, root_state: str, max_simulations: int = 16
    ) -> Tuple[bool, List[str], MCTSNode]:
        root = MCTSNode(state=root_state, prior=1.0)
        root.visit_count = 1

        for _ in range(max_simulations):
            leaf = self.select(root)
            child, val = self.expand_and_evaluate(leaf)
            self.backpropagate(child, val)
            if child.is_solved:
                return True, child.extract_action_trace(), root

            curr = child
            for _ in range(2):
                if curr.is_terminal or curr.is_solved:
                    break
                nxt_child, nxt_val = self.expand_and_evaluate(curr)
                self.backpropagate(nxt_child, nxt_val)
                if nxt_child.is_solved:
                    return True, nxt_child.extract_action_trace(), root
                curr = nxt_child

        curr = root
        while curr.children:
            curr = max(curr.children.values(), key=lambda c: (c.visit_count, c.q_value))
        return curr.is_solved, curr.extract_action_trace(), root

    @staticmethod
    def to_ascii_tree(root: MCTSNode) -> str:
        lines = []

        def _walk(node: MCTSNode, prefix: str = "", is_last: bool = True):
            conn = "└── " if is_last else "├── "
            if node.parent is None:
                lines.append(f"ROOT: {node.state}  [Visits N={node.visit_count}]")
            else:
                tag = "SOLVED" if node.is_solved else ("DEAD-END" if node.prm_value < 0.25 else "EXPLORED")
                lines.append(
                    f"{prefix}{conn}{node.action}  |  "
                    f"Prior P={node.prior:.2f}  |  PRM V={node.prm_value:.2f}  |  "
                    f"Q={node.q_value:.2f} (N={node.visit_count})  [{tag}]"
                )
            child_prefix = prefix + ("    " if is_last else "│   ")
            children = list(node.children.values())
            for i, c in enumerate(children):
                _walk(c, child_prefix, i == len(children) - 1)

        _walk(root)
        return "\n".join(lines)

    @staticmethod
    def to_mermaid(root: MCTSNode) -> str:
        lines = ["graph TD"]
        counter = [0]
        node_ids: Dict[int, str] = {}

        def visit(n: MCTSNode):
            nid = f"N{counter[0]}"
            counter[0] += 1
            node_ids[id(n)] = nid
            label = n.action if n.action else n.state
            status = "SOLVED" if n.is_solved else f"Q={n.q_value:.2f}, N={n.visit_count}"
            clean = label.replace('"', "'").replace("<", "&lt;").replace(">", "&gt;")
            lines.append(f'    {nid}["{clean}<br/>({status})"]')
            for child in n.children.values():
                visit(child)
                cid = node_ids[id(child)]
                lines.append(f'    {nid} -->|"P={child.prior:.2f}, V={child.prm_value:.2f}"| {cid}')

        visit(root)
        return "\n".join(lines)