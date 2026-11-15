import io.shiftleft.codepropertygraph.generated.PropertyNames
import io.shiftleft.codepropertygraph.generated.nodes._
import io.shiftleft.semanticcpg.language._
import io.shiftleft.semanticcpg.utils.MemberAccess
import org.apache.commons.text.StringEscapeUtils

import java.util.Optional
import scala.collection.immutable.HashMap
import scala.collection.mutable
import scala.language.postfixOps

import org.json4s.native.Serialization.{write, writePretty}
import org.json4s.{CustomSerializer, Extraction, Formats}

object GraphSerializer {

  private val charLimit = 5000

  case class Graph(
      vertices: List[StoredNode],
      edges: List[LineEdge],
      subgraph: Map[String, Seq[StoredNode]] =
        HashMap.empty[String, Seq[StoredNode]]
  ) {
    def ++(other: Graph): Graph = {
      Graph(
        (this.vertices ++ other.vertices).distinct,
        (this.edges ++ other.edges).distinct
      )
    }

    // merge same line nodes
    def mergeNodesWithSameLineNumber(codeLines: List[String]): Graph = {
      val lineNumberToNodes = vertices
        .filter(v => v.propertiesMap.get(PropertyNames.LineNumber) != null)
        .groupBy(_.propertiesMap.get(PropertyNames.LineNumber))

      // Map nodes by merging nodes and updating properties
      val mergedVertices = lineNumberToNodes.map {
        case (lineNumberOpt, nodes) =>
          val representativeNode = nodes.head

          lineNumberOpt match {
            case lineNumber: Int =>
              // Check if the line number is a valid index for codeLines
              if (lineNumber > 0 && lineNumber <= codeLines.length) {
                val code = codeLines(lineNumber - 1)
                representativeNode.propertiesMap.getOrDefault(
                  PropertyNames.Code,
                  code
                )
              }
            case _ =>
              println(s"Unexpected line number type: $lineNumberOpt")
              None
          }

          representativeNode
      }.toList

      // Update edges to use line numbers
      val updatedEdges = edges.map { edge =>
        LineEdge(
          srcLineNumber = edge.srcLineNumber,
          dstLineNumber = edge.dstLineNumber,
          label = edge.label,
          edgeType = edge.edgeType,
          srcCharacterPositions = edge.srcCharacterPositions,
          dstCharacterPositions = edge.dstCharacterPositions
        )
      }.distinct

      Graph(mergedVertices, updatedEdges, subgraph)
    }
  }

  case class LineEdge(
      srcLineNumber: Int,
      dstLineNumber: Int,
      label: String = "",
      edgeType: String = "",
      srcCharacterPositions: Option[(Int, Int)] =
        None, // (start, end) character positions
      dstCharacterPositions: Option[(Int, Int)] =
        None // for token-level granularity
  )

  case class Node(line: Int, code: String)
  case class Edge(
      from: Int,
      to: Int,
      edgeType: String,
      varName: Option[String] = None
  )
  case class JsonGraph(nodes: Seq[Node], edges: Seq[Edge])

  def dotGraph(
      root: Option[AstNode] = None,
      graph: Graph,
      codeLines: List[String],
      withEdgeTypes: Boolean = false
  ): String = {
    val mergedGraph = graph.mergeNodesWithSameLineNumber(codeLines)
    val sb = root match {
      case Some(r) => namedGraphBegin(r)
      case None    => defaultGraphBegin()
    }
    val nodeStrings = mergedGraph.vertices.map(v => nodeToDot(v, codeLines))
    val edgeStrings = mergedGraph.edges.map(e => edgeToDot(e, withEdgeTypes))
    val subgraphStrings = mergedGraph.subgraph.zipWithIndex.map {
      case ((subgraph, nodes), idx) =>
        nodesToSubGraphs(subgraph, nodes, idx)
    }
    sb.append((nodeStrings ++ edgeStrings ++ subgraphStrings).mkString("\n"))
    graphEnd(sb)
  }

  def jsonGraph(graph: Graph, codeLines: List[String]): String = {
    val mergedGraph = graph.mergeNodesWithSameLineNumber(codeLines)

    val nodes = mergedGraph.vertices
      .map { node =>
        val lineNumber =
          node.propertyOption(PropertyNames.LineNumber).get.asInstanceOf[Int]
        val code = codeLines(lineNumber - 1).stripSuffix("{")
        Node(line = lineNumber, code = code.trim())
      }
      .sortBy(_.line)

    val edges = mergedGraph.edges
      .map { edge =>

        val optionalVar: Option[String] = edge.label match {
          case s if s.nonEmpty => Some(edge.label)
          case _               => None
        }

        Edge(
          from = edge.srcLineNumber,
          to = edge.dstLineNumber,
          edgeType = edge.edgeType,
          varName = optionalVar
        )
      }
      .sortBy(_.from)

    val jsonGraph = JsonGraph(nodes = nodes, edges = edges)

    implicit val formats: Formats = org.json4s.DefaultFormats
    write(jsonGraph)
  }

  def nodeLinkGraph(
      graph: Graph,
      codeLines: List[String],
      vulnerableRanges: List[(Int, Int)] = List.empty
  ): String = {
    val mergedGraph = graph.mergeNodesWithSameLineNumber(codeLines)
    val vertexLineNumbers = mergedGraph.vertices.map { node =>
      node.propertyOption(PropertyNames.LineNumber).get.asInstanceOf[Int]
    }.toSet

    val nodes = mergedGraph.vertices.map { node =>
      val lineNumber =
        node.propertyOption(PropertyNames.LineNumber).get.asInstanceOf[Int]
      val code = codeLines(lineNumber - 1) // .stripSuffix("{")
      Map(
        "id" -> lineNumber,
        "code" -> code.trim() // .trim()
      )
    }

    val links = mergedGraph.edges
      .filter(edge =>
        vertexLineNumbers.contains(edge.srcLineNumber) &&
          vertexLineNumbers.contains(edge.dstLineNumber)
      )
      .map { edge =>
      Map(
        "source" -> edge.srcLineNumber,
        "target" -> edge.dstLineNumber,
        "key" -> s"${edge.srcLineNumber}-${edge.dstLineNumber}-${edge.edgeType}-${edge.label}",
        "type" -> edge.edgeType,
        "label" -> edge.label,
        "srcCharacters" -> edge.srcCharacterPositions
          .map(p => Map("start" -> p._1, "end" -> p._2))
          .getOrElse(Map()),
        "dstCharacters" -> edge.dstCharacterPositions
          .map(p => Map("start" -> p._1, "end" -> p._2))
          .getOrElse(Map())
      )
    }

    val data = Map(
      "directed" -> true,
      "multigraph" -> true,
      "graph" -> Map(
        "vulnerableRanges" -> vulnerableRanges.map { case (start, end) =>
          Map("start" -> start, "end" -> end)
        }
      ),
      "nodes" -> nodes,
      "links" -> links
    )

    implicit val formats: Formats = org.json4s.DefaultFormats
    writePretty(data)
  }

  private def namedGraphBegin(root: AstNode): mutable.StringBuilder = {
    val sb = new mutable.StringBuilder
    val name = StringEscapeUtils.escapeHtml4(root match {
      case method: Method => method.name
      case _              => ""
    })
    sb.append(s"""digraph "$name" {  \n""")
  }

  private def defaultGraphBegin(): mutable.StringBuilder = {
    val sb = new mutable.StringBuilder
    val name = "CPG"
    sb.append(s"""digraph "$name" {  \n""")
  }

  private def limit(str: String): String = if (str.length > charLimit) {
    s"${str.take(charLimit - 3)}..."
  } else {
    str
  }

  private def stringRepr(
      vertex: StoredNode,
      codeLines: List[String]
  ): String = {
    val maybeLineNo: Option[Any] =
      vertex.propertyOption(PropertyNames.LineNumber)
    val code = maybeLineNo match {
      case Some(lineNo: Int) => codeLines(lineNo - 1).stripSuffix("{")
      case _                 => ""
    }
    val escapedCode = StringEscapeUtils.escapeHtml4(code)

    escapedCode + maybeLineNo.map(lineNo => s"<SUB>$lineNo</SUB>").getOrElse("")
  }

  private def toCfgNode(node: StoredNode): CfgNode = {
    node match {
      case node: Identifier         => node.parentExpression.get
      case node: MethodRef          => node.parentExpression.get
      case node: Literal            => node.parentExpression.get
      case node: MethodParameterIn  => node.method
      case node: MethodParameterOut => node.method.methodReturn
      case node: io.shiftleft.codepropertygraph.generated.nodes.Call
          if MemberAccess.isGenericMemberAccessName(node.name) =>
        node.parentExpression.get
      case node: io.shiftleft.codepropertygraph.generated.nodes.CallRepr => node
      case node: MethodReturn                                            => node
      case node: Expression                                              => node
    }
  }

  private def nodeToDot(node: StoredNode, codeLines: List[String]): String = {
    val lineNumber =
      node.propertyOption(PropertyNames.LineNumber).get.asInstanceOf[Int]
    val label = stringRepr(node, codeLines)

    val fillColor =
      if (label.contains("vul-start") || label.contains("vul-end")) {
        "yellow"
      } else {
        "white"
      }

    s""""$lineNumber" [label = <$label> style=filled fillcolor="$fillColor"]""".stripMargin
  }

  private def edgeToDot(
      edge: GraphSerializer.LineEdge,
      withEdgeTypes: Boolean
  ): String = {
    val edgeLabel = if (withEdgeTypes) {
      edge.edgeType + (if (edge.label.nonEmpty)
                         (": " + StringEscapeUtils.escapeHtml4(edge.label))
                       else "")
    } else {
      StringEscapeUtils.escapeHtml4(edge.label)
    }

    val color = edge.edgeType match {
      case "DDG" => "blue"
      case "CDG" => "orange"
      case "CFG" => "red"
      case "REF" => "gray"
      case _     => "black"
    }

    val labelStr = Some(s""" [ label = "$edgeLabel" color = "$color"] """)
      .filter(_ => edgeLabel.nonEmpty)
      .getOrElse("")
    s"""  "${edge.srcLineNumber}" -> "${edge.dstLineNumber}" """ + labelStr
  }

  def nodesToSubGraphs(
      subgraph: String,
      children: Seq[StoredNode],
      idx: Int
  ): String = {
    val escapedName = StringEscapeUtils.escapeHtml4(subgraph)
    val childString = children.map { c => s"    \"${c.id()}\";" }.mkString("\n")
    s"""  subgraph cluster_$idx {
       |$childString
       |    label = "$escapedName";
       |  }
       |""".stripMargin
  }

  private def graphEnd(sb: mutable.StringBuilder): String = {
    sb.append("\n}\n")
    sb.toString
  }
}
