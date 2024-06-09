import java.util.HashMap;
import java.util.Map;

class DecisionNode {
    String feature; // Feature to split on
    DecisionNode left; // Left child node for false
    DecisionNode right; // Right child node for true
    String decision; // Final decision to be made

    DecisionNode(String feature, DecisionNode left, DecisionNode right, String decision) { // constructor for Decision
        this.feature = feature;
        this.left = left;
        this.right = right;
        this.decision = decision;
    }
}

class DecisionTree {
    private DecisionNode root;

    // create first node and set it to empty
    public DecisionTree() {
        this.root = null;
    }

    public void buildDecisionTree() {
        // Hardcoded decision tree

        // Feature check for whether it's raining
        DecisionNode rainNode = new DecisionNode("Is it raining?", root, root, null);

        // Decision for rain feature
        DecisionNode decisionYes = new DecisionNode(null, null, null, "Take an umbrella");
        DecisionNode decisionNo = new DecisionNode(null, null, null, "Do not take an umbrella");

        rainNode.left = decisionYes; // left equals yes, take umbrella
        rainNode.right = decisionNo; // right equals no, do not take umbrella

        this.root = rainNode;
    }

    private String traverseDecisionTree(DecisionNode node, boolean isFeatureTrue) {
        if (node.decision != null) {
            return node.decision; // lead node will return decision
        }

        // Traverse left or right depending on feature value
        if (isFeatureTrue) {
            return traverseDecisionTree(node.left, isFeatureTrue);
        } else {
            return traverseDecisionTree(node.right, isFeatureTrue);
        }
    }

    public String makeDecision(boolean isRaining) {
        return traverseDecisionTree(root, isRaining);
    }

}

public class DecisionTrees {
    public static void main(String[] args) {
        DecisionTree decisionTree = new DecisionTree();
        decisionTree.buildDecisionTree();

        // Test
        boolean isRaining = false;
        String decision = decisionTree.makeDecision(isRaining);

        System.out.println("Decision: " + decision);
    }

}
