# The neighbouring class of check, configuration (a): a declarative per-call
# policy with NO data-flow input.
#
# This is what a real policy engine is given in the common deployment: the call
# (tool name and arguments) and the operator's static facts (which tools are
# egress sinks, which destinations are approved, bounds on amounts). It is the
# same information the axor taxonomy declares, minus the per-value provenance
# axor derives itself.
#
# Decision: deny if a declared predicate over an argument fails; otherwise allow.
#
#   opa eval -d call_only.rego -i input.json 'data.neighbor.decision'
package neighbor

import rego.v1

default allow := true

# An enum predicate over an argument: the value must be in the approved set.
deny contains reason if {
	some pred in input.config.value_policies[input.tool]
	pred.kind == "enum"
	value := input.args[pred.arg]
	not value in pred.allowed
	reason := sprintf("arg %v=%v not in the approved set", [pred.arg, value])
}

# A numeric bound over an argument.
deny contains reason if {
	some pred in input.config.value_policies[input.tool]
	pred.kind == "numeric_range"
	value := input.args[pred.arg]
	not within(value, pred.lo, pred.hi)
	reason := sprintf("arg %v=%v outside [%v, %v]", [pred.arg, value, pred.lo, pred.hi])
}

within(v, lo, hi) if {
	is_number(v)
	v >= lo
	v <= hi
}

decision := {
	"allowed": count(deny) == 0,
	"reasons": deny,
	"basis": "call + static operator facts (no data-flow labels)",
}
