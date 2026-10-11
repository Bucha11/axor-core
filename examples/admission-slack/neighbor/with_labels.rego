# The neighbouring class of check, configuration (b): the same declarative
# per-call policy, but handed axor's own observations as input.
#
# The labels (`input.labels`) are exactly what the kernel derives and the gates
# read: for each argument, the origin class of the value (its causal sources and
# trusted origin) plus the session's confidentiality-floor boolean. Giving them to
# the engine is the fair comparison: with the same observations, does a different
# engine separate the pair?
#
# The rules below are the AgentDojo-relevant subset of the axor gate sequence,
# transcribed: value predicates, then integrity (an untrusted-derived driving
# value into a declared sink), then the confidentiality floor, with the enum
# supersession on fully enum-covered driving arguments.
#
#   opa eval -d with_labels.rego -i input.json 'data.neighbor_labels.decision'
package neighbor_labels

import rego.v1

default allow := true

sink if input.tool in input.config.egress_sinks

sink if input.tool in input.config.integrity_sinks

egress if input.tool in input.config.egress_sinks

drivers := object.get(input.config.driving_args, input.tool, [])

# Enum supersession (T4): every driving argument present and covered by an enum
# over a closed operator-declared set carries the integrity axis itself.
enum_covered contains pred.arg if {
	some pred in input.config.value_policies[input.tool]
	pred.kind == "enum"
}

superseded if {
	count(drivers) > 0
	every d in drivers {
		d in enum_covered
		input.args[d]
	}
}

tainted_driver if {
	some d in drivers
	input.labels[d].tainted
}

# With no declared driving arguments the decision falls back to the whole call,
# exactly as the kernel's whole-args default does.
tainted_driver if {
	count(drivers) == 0
	some _, label in input.labels
	label.tainted
}

deny contains reason if {
	some pred in input.config.value_policies[input.tool]
	pred.kind == "enum"
	value := input.args[pred.arg]
	not value in pred.allowed
	reason := sprintf("arg %v=%v not in the approved set", [pred.arg, value])
}

deny contains reason if {
	some pred in input.config.value_policies[input.tool]
	pred.kind == "numeric_range"
	value := input.args[pred.arg]
	not within(value, pred.lo, pred.hi)
	reason := sprintf("arg %v=%v outside [%v, %v]", [pred.arg, value, pred.lo, pred.hi])
}

deny contains "integrity: untrusted-derived driving value into a declared sink" if {
	sink
	tainted_driver
	not superseded
}

deny contains "confidentiality floor: egress while a secret read is outstanding" if {
	egress
	input.floor_active
}

within(v, lo, hi) if {
	is_number(v)
	v >= lo
	v <= hi
}

decision := {
	"allowed": count(deny) == 0,
	"reasons": deny,
	"basis": "call + static operator facts + axor's per-value origin labels",
}
