"""
Visualize neurons as trees
"""

import json
import numpy as np


def make_treedata(contrs, tallies, by='feat_corr', root_collapsed=False, collapsed=True, root_name='resnet18'):
    # Loop in reverse order
    contrs_rev = contrs[::-1]
    tallies_rev = tallies[::-1]
    tds = []
    for unit in range(contrs_rev[0][by]['contr'][0].shape[0]):
        td = make_treedata_rec(unit, contrs_rev, tallies_rev, root_name, by=by, collapsed=collapsed)
        tds.append(td)
    if root_collapsed:
        k = '_children'
    else:
        k = 'children'
    return {
        'name': root_name,
        'parent': 'null',
        k: tds
    }


def make_treedata_rec(unit, contrs, tallies, parent_name, parent_weight=None, by='feat_corr', collapsed=True):
    # Loop in reverse order
    this_contr, _ = contrs[0][by]['contr']
    this_weight = contrs[0][by]['weight']
    this_tally = tallies[0]
    this_name = f'{unit}-{this_tally[unit + 1]}'
    if parent_weight is not None:
        this_name = f'{this_name} ({parent_weight:.2f})'
    this = {
        'name': this_name,
        'parent': parent_name,
    }
    if this_contr is not None:
        if collapsed:
            k = '_children'
        else:
            k = 'children'
        this[k] = [
            make_treedata_rec(u, contrs[1:], tallies[1:], this_name, parent_weight=this_weight[unit, u])
            for u in np.where(this_contr[unit])[0]
        ]
    return this


EXAMPLE_TREEDATA = """
[
  {
    "name": "Top Level",
    "parent": "null",
    "children": [
      {
        "name": "Level 2: A",
        "parent": "Top Level",
        "children": [
          {
            "name": "Son of A",
            "parent": "Level 2: A"
          },
          {
            "name": "Daughter of A",
            "parent": "Level 2: A"
          }
        ]
      },
      {
        "name": "Level 2: B",
        "parent": "Top Level"
      }
    ]
  }
];
"""

TREESTYLE = r"""
<style>
    .node {
            cursor: pointer;
    }

    .node circle {
      fill: #fff;
      stroke: steelblue;
      stroke-width: 3px;
    }

    .node text {
      font: 12px sans-serif;
    }

    .link {
      fill: none;
      stroke: #ccc;
      stroke-width: 2px;
    }
</style>
"""

TREESCRIPT = r"""
<script>

// ************** Generate the tree diagram     *****************
var margin = {top: 20, right: 120, bottom: 20, left: 120},
    width = 1960 - margin.right - margin.left,
    height = 5000 - margin.top - margin.bottom;

var i = 0,
    duration = 750;

var root = treeData;
root.x0 = height / 2;
root.y0 = 0;

var tree = d3.layout.tree()
    .size([height, width]);

var diagonal = d3.svg.diagonal()
    .projection(function(d) { return [d.y, d.x]; });

var svg = d3.select("body").append("svg")
    .attr("width", width + margin.right + margin.left)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Toggle children on click.
var update = function(source) {

  var click = function(d) {
    if (d.children) {
      d._children = d.children;
      d.children = null;
    } else {
      d.children = d._children;
      d._children = null;
    }
    update(d);
  }

  // Compute the new tree layout.
  var nodes = tree.nodes(root).reverse(),
      links = tree.links(nodes);

  // Normalize for fixed-depth.
  nodes.forEach(function(d) { d.y = d.depth * 180; });

  // Update the nodes…
  var node = svg.selectAll("g.node")
      .data(nodes, function(d) { return d.id || (d.id = ++i); });

  // Enter any new nodes at the parent's previous position.
  var nodeEnter = node.enter().append("g")
      .attr("class", "node")
      .attr("transform", function(d) { return "translate(" + source.y0 + "," + source.x0 + ")"; })
      .on("click", click);

  nodeEnter.append("circle")
      .attr("r", 1e-6)
      .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });

  nodeEnter.append("text")
      .attr("x", function(d) { return d.children || d._children ? -13 : 13; })
      .attr("dy", ".35em")
      .attr("text-anchor", function(d) { return d.children || d._children ? "end" : "start"; })
      .text(function(d) { return d.name; })
      .style("fill-opacity", 1e-6);

  // Transition nodes to their new position.
  var nodeUpdate = node.transition()
      .duration(duration)
      .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

  nodeUpdate.select("circle")
      .attr("r", 10)
      .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });

  nodeUpdate.select("text")
      .style("fill-opacity", 1);

  // Transition exiting nodes to the parent's new position.
  var nodeExit = node.exit().transition()
      .duration(duration)
      .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
      .remove();

  nodeExit.select("circle")
      .attr("r", 1e-6);

  nodeExit.select("text")
      .style("fill-opacity", 1e-6);

  // Update the links…
  var link = svg.selectAll("path.link")
      .data(links, function(d) { return d.target.id; });

  // Enter any new links at the parent's previous position.
  link.enter().insert("path", "g")
      .attr("class", "link")
      .attr("d", function(d) {
        var o = {x: source.x0, y: source.y0};
        return diagonal({source: o, target: o});
      });

  // Transition links to their new position.
  link.transition()
      .duration(duration)
      .attr("d", diagonal);

  // Transition exiting nodes to the parent's new position.
  link.exit().transition()
      .duration(duration)
      .attr("d", function(d) {
        var o = {x: source.x, y: source.y};
        return diagonal({source: o, target: o});
      })
      .remove();

  // Stash the old positions for transition.
  nodes.forEach(function(d) {
    d.x0 = d.x;
    d.y0 = d.y;
  });
}

update(root);

d3.select(self.frameElement).style("height", "500px");

</script>
"""
