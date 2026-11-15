// @Time: 2025.4.29 16:00
// @Author: Noah Frontzkowski
// @Desc: RUN > joern --script gen_vdg.scala --param filepaths=test.c,test2.c --param saveprefix=tmp
import scala.io.Source
import scala.util.{Try, Using, Failure, Success}
import java.util.UUID;
import scala.collection.mutable.ListBuffer
import java.nio.file.{Files, Paths}
import java.nio.file.NoSuchFileException
import java.io.{FileWriter, BufferedWriter}
import java.lang.ProcessHandle
import scala.concurrent.{Future, ExecutionContext}
import scala.concurrent.ExecutionContext.Implicits.global

import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.generated.edges.*
import io.shiftleft.semanticcpg.language._
import io.joern.dataflowengineoss.DefaultSemantics
import io.joern.dataflowengineoss.semanticsloader.Semantics
import io.shiftleft.codepropertygraph.generated.nodes.{Method, Block}
import io.joern.dataflowengineoss.dotgenerator.DdgGenerator
import io.shiftleft.semanticcpg.dotgenerator.{AstGenerator, CdgGenerator, CfgGenerator}
import io.shiftleft.semanticcpg.dotgenerator.DotSerializer.{Edge, Graph}
import io.shiftleft.semanticcpg.language.dotextension.Shared


//> using dep org.apache.commons:commons-text:1.11.0
//> using dep org.zeromq:jeromq:0.5.3

import org.apache.commons.text.StringEscapeUtils
import org.zeromq.{SocketType, ZMQ, ZContext}

@main def exec(paramsFilePath: String, zmqEndpoint: Option[String] = None, lang: Option[String] = Some("cpp")): Unit = {

  val (context, socket) = zmqEndpoint.map { endpoint =>
    val ctx = new ZContext()
    val sock = ctx.createSocket(SocketType.PAIR)
    sock.setLinger(1000)
    sock.connect(endpoint)
    (ctx, sock)
  }.getOrElse((null, null))

  def sendMessage(message: String): Unit = {
    zmqEndpoint match {
      case Some(_) => socket.send(message.getBytes(ZMQ.CHARSET), 0)
      case None => println(message)
    }
  }

  val params = Using(Source.fromFile(paramsFilePath)) { source =>
    source
      .getLines()
      .map { line =>
        val Array(key, value) = line.split("=", 2)
        key -> value.replace("\"", "")
      }
      .toMap
  }.getOrElse {
    throw new IllegalArgumentException("Failed to read parameters file.")
  }

  val filepaths = params("filepaths")
  val saveprefixes = params("saveprefixes")
  val statsbasepaths = params.getOrElse("statsbasepaths", "")

  val filepathsList =
    if (filepaths.contains(",")) filepaths.split(",").toList
    else List(filepaths)
  val saveprefixesList =
    if (saveprefixes.contains(",")) saveprefixes.split(",").toList
    else List(saveprefixes)
  val statsbasepathsList =
    if (statsbasepaths.nonEmpty && statsbasepaths.contains(","))
      statsbasepaths.split(",").toList
    else if (statsbasepaths.nonEmpty) List(statsbasepaths)
    else List()

  require(
    filepathsList.length == saveprefixesList.length,
    "filepaths and saveprefixes must have the same length"
  )
  if (statsbasepathsList.nonEmpty) {
    require(
      filepathsList.length == statsbasepathsList.length,
      "filepaths and statsbasepaths must have the same length if statsbasepaths is provided"
    )
  }

  switchWorkspace(".sequelizer_tmp/workspace_" + ProcessHandle.current().pid())
  Try(workspace.reset).toOption

  val result = Try {
    def importWithRetry(filepath: String, maxRetries: Int = 3): Boolean = {
      def attempt(retries: Int): Boolean = {
        // there is a strange Joern bug; in rare instances Joern throws a "NoSuchFileException: project.json",
        // to avoid subsequent retries using the same project folder (and therefore encounter the same issue)
        // we use random UUIDs for project names
        Try(importCpg(filepath, UUID.randomUUID().toString())) match {
          case Success(Some(result)) =>
            true
          case Success(_) =>
            sendMessage(
              s"""{"type":"warning","message":"CPG import returned null for $filepath"}"""
            )
            false
          case Failure(exception) if retries > 0 =>
            sendMessage(
              s"""{"type":"warning","message":"CPG import failed. Error: ${exception.getMessage}. Retrying ..."}"""
            )
            Try(workspace.reset)
            Thread.sleep(1000) // Wait for 1 second before retrying
            attempt(retries - 1)
          case Failure(exception) =>
            sendMessage(
              s"""{"type":"warning","message":"Failed to import CPG: ${exception.getMessage}"}"""
            )
            // throw exception // Propagate other exceptions immediately
            false
        }
      }
      attempt(maxRetries)
    }

    def findParentStructure(node: AstNode, visited: Set[AstNode] = Set.empty): Option[StoredNode] = {
      if (visited.contains(node)) {
        None // cycle
      } else if (node.label == "CONTROL_STRUCTURE" || node.label == "METHOD") {
        Some(node.asInstanceOf[StoredNode])
      } else {
        node.inAst.headOption.flatMap(parent =>
          findParentStructure(parent.asInstanceOf[AstNode], visited + node)
        )
      }
    }


    sendMessage(
      s"""{"type":"debug","message":"Bootstrap done. Starting VDG generation."}"""
    )

    try {
      filepathsList.zipWithIndex.foreach { case (filepath, idx) =>
        val saveprefix = saveprefixesList(idx)
        val statsbasepath =
          if (statsbasepathsList.nonEmpty) statsbasepathsList(idx) else ""
        val outputFile = saveprefix + "_node_link.json"

        if (Files.exists(Paths.get(outputFile))) {
          sendMessage(
            s"""{"type":"nop","file":"$outputFile", "cpg_file": "$filepath", "info":"VDG already exists!"}"""
          )
          sendMessage(
            s"""{"type":"progress","current":${idx + 1},"total":${filepathsList.length}}"""
          )
        } else {
          // import CPG
          if (importWithRetry(filepath)) {
            // constants
            val redundantEdgeTypes = Seq("REF", "AST")

            // OSS data flow analysis layer is run automatically by Joern
            // run.ossdataflow

            // generate and export PDG
            val codeFile = filepath.replaceAll("\\.cpg\\.bin\\.zip$", "." + lang.getOrElse("cpp"))
            val lines = scala.io.Source.fromFile(codeFile).getLines().toList
            val markerLines = lines.zipWithIndex.collect {
              case (line, index)
                  if line.contains("vul-start") || line.contains("vul-end") =>
                index + 1
            }
            val candidateMethods = cpg.method
              .filterNot(mtd =>
                mtd.isExternal || mtd.signature.isEmpty || mtd.lineNumber.isEmpty || mtd.lineNumberEnd.isEmpty
              )
              .l
            val methodOption = if (markerLines.nonEmpty) {
              candidateMethods
                .filter(method =>
                  markerLines.forall(line =>
                    line >= method.lineNumber.get.toInt &&
                      line <= method.lineNumberEnd.get.toInt
                  )
                )
                .sortBy(method => method.lineNumberEnd.get.toInt - method.lineNumber.get.toInt)
                .headOption
            } else {
              candidateMethods.headOption
            }

            methodOption match {
              case Some(method) =>
                val taggedVulnerableLines = getVulnerableLines(method, Some(lines))

                val ddg = new DdgGenerator().generate(method)
                val cdg = new CdgGenerator().generate(method)
                val cfg = new CfgPlusGenerator().generate(method)

                val pdg = ddg ++ cdg ++ cfg
                val vulnerableLinesForRefs = alignVulnerableRangesToVertices(
                  taggedVulnerableLines,
                  pdg.vertices.distinct,
                  method
                )
                val vulnerableIndices = vulnerableLinesForRefs.flatMap {
                  case (start, end) => List(start, end)
                }.toSet

                val pdgEdges = pdg.edges
                  .filter { edge =>
                    val srcNode = edge.src.asInstanceOf[CfgNode]
                    val dstNode = edge.dst.asInstanceOf[CfgNode]
                    srcNode.lineNumber.isDefined && dstNode.lineNumber.isDefined &&
                    !srcNode.isInstanceOf[MethodReturn] && !dstNode
                      .isInstanceOf[MethodReturn] &&
                    !(edge.edgeType == "DDG" && edge.label.isEmpty)
                  }
                  .map { edge =>
                    val srcNode = edge.src.asInstanceOf[CfgNode]
                    val dstNode = edge.dst.asInstanceOf[CfgNode]

                    val srcLineNumber = srcNode.lineNumber.get.asInstanceOf[Int]
                    val dstLineNumber = dstNode.lineNumber.get.asInstanceOf[Int]
                    val variableLabel = removeVulMarkers(edge.label)

                    val (srcLine, srcCharacterPos) =
                      getCharacterPosition(srcNode, lines, sendMessage)

                    val (dstLine, dstCharacterPos) =
                      getCharacterPosition(dstNode, lines, sendMessage)


                    GraphSerializer.LineEdge(
                      srcLineNumber = srcLine,
                      dstLineNumber = dstLine,
                      label = variableLabel,
                      edgeType = edge.edgeType,
                      srcCharacterPositions = srcCharacterPos,
                      dstCharacterPositions = dstCharacterPos
                    )
                  }
                val refNodes: ListBuffer[StoredNode] = ListBuffer()

                val refEdges = method.graph.allEdges
                  .filter(edge => edge.isInstanceOf[Ref])
                  .map(edge => edge.asInstanceOf[Ref])
                  .filter(ref =>
                    ref.src.propertiesMap.containsKey(
                      PropertyNames.LineNumber
                    ) && ref.dst.propertiesMap.containsKey(
                      PropertyNames.LineNumber
                    )
                  )
                  .filter(ref =>
                    vulnerableIndices.contains(
                      ref.dst
                        .property(PropertyNames.LineNumber)
                        .asInstanceOf[Int]
                    )
                  ) // only include vulnerable REF nodes
                  .filter(ref =>
                    !pdgEdges.exists(pdgEdge =>
                      pdgEdge.dstLineNumber == ref.dst
                        .property(PropertyNames.LineNumber)
                        .asInstanceOf[Int]
                    )
                  ) // exclude ref edges that lead to nodes that are already included in the PDG
                  .flatMap(ref => {
                    val srcNode = ref.src.asInstanceOf[StoredNode]
                    val dstNode = ref.dst.asInstanceOf[StoredNode] // REF node

                    val cfgParent = if (!method.ast.contains(dstNode.asInstanceOf[AstNode])) {
                      method
                    } else {
                      findParentStructure(dstNode.asInstanceOf[AstNode])
                        .getOrElse(method) // fallback to method; sometimes Joern has cycles in the AST...
                    }

                    val parentLineNumber =
                      cfgParent
                        .property(PropertyNames.LineNumber)
                        .asInstanceOf[Int]

                    val srcLineNumber = ref.src
                      .property(PropertyNames.LineNumber)
                      .asInstanceOf[Int]
                    val dstLineNumber = ref.dst
                      .property(PropertyNames.LineNumber)
                      .asInstanceOf[Int]

                    refNodes += cfgParent
                    refNodes += srcNode
                    refNodes += dstNode

                    val varName = dstNode.propertiesMap
                      .get(PropertyNames.Name)
                      .asInstanceOf[String]

                    val (parentLine, refParentCharacterPos) =
                      getCharacterPosition(cfgParent, lines, sendMessage)
                    val (srcLine, srcCharacterPos) =
                      getCharacterPosition(srcNode, lines, sendMessage)
                    val (dstLine, dstCharacterPos) =
                      getCharacterPosition(dstNode, lines, sendMessage)

                    List(
                      GraphSerializer.LineEdge(
                        dstLine,
                        srcLine,
                        label = varName,
                        edgeType = "REF",
                        srcCharacterPositions = dstCharacterPos,
                        dstCharacterPositions = srcCharacterPos
                      ),
                      GraphSerializer.LineEdge(
                        parentLine,
                        dstLine,
                        label = "",
                        edgeType = "AST",
                        srcCharacterPositions = refParentCharacterPos,
                        dstCharacterPositions = dstCharacterPos
                      )
                    )
                  })

                var cpgWithRefEdges = pdgEdges ++ refEdges
                cpgWithRefEdges =
                  removeRedundantEdges(cpgWithRefEdges, redundantEdgeTypes: _*)

                // remove recursive token-level edges
                cpgWithRefEdges = cpgWithRefEdges.filter(edge =>
                  edge.srcLineNumber != edge.dstLineNumber ||
                    edge.srcCharacterPositions != edge.dstCharacterPositions
                )
                // val pdgForwardEdges = pdgRefEdges.filter(edge => edge.srcLineNumber < edge.dstLineNumber)

                // val allVertices = pdg.vertices.distinct
                val allVertices = (pdg.vertices ++ refNodes).distinct
                val vulnerableLines = alignVulnerableRangesToVertices(
                  taggedVulnerableLines,
                  allVertices,
                  method
                )

                // bi-directional slicing on vulnerable nodes
                val vulnerableNodes = allVertices.filter(v =>
                  vulnerableLines.exists { case (start, end) =>
                    val line =
                      v.property(PropertyNames.LineNumber).asInstanceOf[Int]
                    line >= start && line <= end
                  }
                )

                val forwardSlices = vulnerableNodes.flatMap(node =>
                  forwardSlice(node, cpgWithRefEdges, allVertices)
                )
                val backwardSlices = vulnerableNodes.flatMap(node =>
                  backwardSlice(node, cpgWithRefEdges, allVertices)
                )

                val slicedVertices = (forwardSlices ++ backwardSlices).distinct

                val slicedEdges = cpgWithRefEdges.filter(edge =>
                  slicedVertices.exists(v =>
                    v.property(PropertyNames.LineNumber)
                      .asInstanceOf[Int] == (edge.srcLineNumber)
                  ) &&
                    slicedVertices.exists(v =>
                      v.property(PropertyNames.LineNumber)
                        .asInstanceOf[Int] == (edge.dstLineNumber)
                    )
                )

                val resultingGraph = if (slicedVertices.isEmpty) {
                  GraphSerializer.Graph(allVertices, cpgWithRefEdges)
                } else {
                  GraphSerializer.Graph(slicedVertices, slicedEdges)
                }

                GraphSerializer.nodeLinkGraph(
                  resultingGraph,
                  lines,
                  vulnerableLines
                ) |> outputFile
                sendMessage(
                  s"""{"type":"result", "cpg_file": "$filepath", "file":"$outputFile"}"""
                )

              case None =>
                sendMessage(
                  s"""{"type":"debug","message":"No method found in the CPG for $filepath"}"""
                )
                writeEmptyGraph(outputFile)
                sendMessage(
                  s"""{"type":"result","cpg_file": "$filepath", "file":"$outputFile"}"""
                )
            }
          } else {
            sendMessage(
              s"""{"type":"warning","message":"Creating empty graph for $filepath due to import failure"}"""
            )
            writeEmptyGraph(outputFile)
            sendMessage(
              s"""{"type":"result","cpg_file": "$filepath","file":"$outputFile"}"""
            )
          }
          sendMessage(
            s"""{"type":"progress","current":${idx + 1},"total":${filepathsList.length}}"""
          )
        }
        // Try(delete).toOption.foreach(_ => System.gc())
        Try(workspace.reset).toOption
      }
      sendMessage("END")

    } catch {
      case e: Exception =>
        sendMessage(
          s"""{"type":"error","message":"${e.getMessage}"}"""
        )
    } finally {
      if (zmqEndpoint.isDefined) {
        socket.close()
        Thread.sleep(1000)
        context.close()
      }

      Try(delete).toOption
    }
  }

  if (statsbasepaths.nonEmpty) {
    // collectCpgStatistics(cpg, statsbasepaths)
  }

  result match {
    case Success(_) =>
      System.exit(0)
    case Failure(exception) =>
      System.err.println(s"Unexpected error: ${exception.getMessage}")
      exception.printStackTrace(System.err)
      System.exit(-1)
  }
}

implicit class StringOps(str: String) {
  def rangeOfIgnoreWhitespace(search: String): Option[(Int, Int)] = {
    val normalizedSearch = search.replaceAll("\\s+", "")
    if (normalizedSearch.isEmpty) return None

    val originalIndices = str.indices.filter(index => !str(index).isWhitespace)
    val normalizedStr = originalIndices.map(str).mkString
    val idx = normalizedStr.indexOf(normalizedSearch)
    if (idx < 0) None
    else Some((originalIndices(idx), originalIndices(idx + normalizedSearch.length - 1)))
  }
}

def getCharacterPosition(
    node: StoredNode,
    lines: List[String],
    sendMessage: String => Unit = _ => ()
): (Int, Option[(Int, Int)]) = {
  val isVirtual =
    node.isInstanceOf[Method] || node
      .isInstanceOf[Block] || node.propertyOption("CODE").isEmpty

  val codeOpt =
    if (isVirtual) node.propertyOption("NAME")
    else node.propertyOption("CODE")

  val startLine = node.property(PropertyNames.LineNumber).asInstanceOf[Int]
  val endLine = node
    .propertyOption(PropertyNames.LineNumberEnd)
    .getOrElse(startLine)

  if (codeOpt.isDefined) {
    var nodeCode = codeOpt.get.toString.trim()

    if (
      nodeCode.contains("\n")
    ) // this should only happen for ill-formatted code (e.g., not defined preprocessor functions)
      nodeCode = nodeCode.split("\n")(
        0
      ) // in some instances, Joern fails to correctly determine the line number of nodes part of multi-line binary expressions

    // Search through the line range
    for (lineNum <- startLine to endLine) {
      if (lineNum <= lines.length) {
        val line = lines(lineNum - 1).trim()
        var characterRange = line.rangeOfIgnoreWhitespace(nodeCode)
        if (characterRange.isDefined) {
          return (lineNum, characterRange)
        } else if (node.isInstanceOf[Local]) { // multiple variable declarations (e.g., int a,b,c,d)
          nodeCode =
            node.property(PropertyNames.Name).asInstanceOf[String].trim()
          characterRange = line.rangeOfIgnoreWhitespace(nodeCode)
          if (characterRange.isDefined) {
            return (lineNum, characterRange)
          }
        }
      }
    }

    /*sendMessage(
      s"""{"type":"debug","message":"${StringEscapeUtils.escapeJson(
          s"""Could not find '$nodeCode' in lines ${startLine}-${endLine}.\n${(startLine to endLine)
              .map(i => s"$i: ${lines(i - 1).trim}")
              .mkString(
                "\n"
              )}.\n ... for node with type '${node.getClass.getSimpleName}' and properties '${node.propertiesMap}'"""
        )}"}"""
    )*/

  }

  /*sendMessage(
    s"""{"type":"debug","message": "No code or name property found for node: $node"}"""
   )*/

  return (
    startLine,
    Some((0, lines(startLine - 1).trim.length - 1))
  ) // return whole line

}

def writeEmptyGraph(outputFile: String): Unit = {
  val emptyGraph = GraphSerializer.Graph(List.empty, List.empty)
  GraphSerializer.nodeLinkGraph(
    emptyGraph,
    List.empty
  ) |> outputFile
}

def removeRedundantEdges(
    graph: List[GraphSerializer.LineEdge],
    redudantTypes: String*
): List[GraphSerializer.LineEdge] = {
  def extractIds(edge: GraphSerializer.LineEdge): (Int, Int) =
    (edge.srcLineNumber, edge.dstLineNumber)
  val groupedEdges = graph.groupBy(extractIds)

  val filteredEdges = groupedEdges.flatMap { case ((src, dst), edges) =>
    val edgeTypesInGroup = edges.map(_.edgeType).distinct
    if (
      edgeTypesInGroup.exists(edgeType => !redudantTypes.contains(edgeType))
    ) {
      // there is a non-redundant edge from src to dst
      // such that we can remove all redundant edges without disconnecting components
      edges.filter(edge => !redudantTypes.contains(edge.edgeType))
    } else {
      // src and dst are connected exclusively by redundant edges, do not disconnect
      edges
    }
  }

  filteredEdges.toList
}

case class VulnerabilityTag(line: Int, isStart: Boolean)

def getVulnerableLinesFromComments(method: Method): List[(Int, Int)] = {
  val methodStartLine = method.lineNumber.map(_.toInt).getOrElse(Int.MinValue)
  val methodEndLine = method.lineNumberEnd.map(_.toInt).getOrElse(Int.MaxValue)

  val vulStartComments = cpg.comment
    .filter(_.code.contains("vul-start"))
    .filter(_.lineNumber.isDefined)
    .map(c => (c, c.lineNumber.get.toInt))
    .toList

  val vulEndComments = cpg.comment
    .filter(_.code.contains("vul-end"))
    .filter(_.lineNumber.isDefined)
    .map(c => (c, c.lineNumber.get.toInt))
    .toList

  processVulnerabilityTags(
    methodStartLine,
    methodEndLine,
    vulStartComments.map(_._2),
    vulEndComments.map(_._2)
  )
}

def getVulnerableLinesFromSource(method: Method, sourceLines: List[String]): List[(Int, Int)] = {
  val methodStartLine = method.lineNumber.map(_.toInt).getOrElse(Int.MinValue)
  val methodEndLine = method.lineNumberEnd.map(_.toInt).getOrElse(Int.MaxValue)

  val tags = sourceLines.zipWithIndex.flatMap { case (line, idx) =>
    val lineNum = idx + 1  // convert 0-based index to 1-based line number

    // Find all occurrences of each tag type
    val startTags = countTagOccurrences(line, "vul-start")
      .map(_ => VulnerabilityTag(lineNum, true))

    val endTags = countTagOccurrences(line, "vul-end")
      .map(_ => VulnerabilityTag(lineNum, false))

    startTags ++ endTags
  }

  val (startLines, endLines) = tags.partition(_.isStart)

  processVulnerabilityTags(
    methodStartLine,
    methodEndLine,
    startLines.map(_.line),
    endLines.map(_.line)
  )
}

private def countTagOccurrences(line: String, tag: String): List[Int] = {
  var pos = 0
  var indices = List[Int]()
  while ({pos = line.indexOf(tag, pos); pos} != -1) {
    indices = pos :: indices
    pos += 1
  }
  indices.reverse
}

private def processVulnerabilityTags(
    methodStartLine: Int,
    methodEndLine: Int,
    startLines: List[Int],
    endLines: List[Int]
): List[(Int, Int)] = {
  val vulStartTags = startLines.filter(line =>
    line >= methodStartLine && line <= methodEndLine
  )
  var vulEndTags = endLines.filter(line =>
    line >= methodStartLine && line <= methodEndLine
  )

  vulStartTags
    .foldLeft(List.empty[(Int, Int)]) {
      case (vulnerableRanges, startLine) =>
        val endTags = vulEndTags.filter(endLine => endLine >= startLine)
        if (endTags.nonEmpty) {
          val endLine = endTags.min // closest end comment
          vulEndTags = vulEndTags.filterNot(_ == endLine)
          vulnerableRanges :+ (startLine, endLine)
        } else {
          vulnerableRanges
        }
    }
    .distinct
}

def alignVulnerableRangesToVertices(
    vulnerableRanges: List[(Int, Int)],
    vertices: List[StoredNode],
    method: Method
): List[(Int, Int)] = {
  val methodStartLine = method.lineNumber.map(_.toInt).getOrElse(Int.MinValue)
  val methodEndLine = method.lineNumberEnd.map(_.toInt).getOrElse(Int.MaxValue)
  val vertexLines = vertices
    .filter(_.propertiesMap.containsKey(PropertyNames.LineNumber))
    .map(_.property(PropertyNames.LineNumber).asInstanceOf[Int])
    .filter(line => line >= methodStartLine && line <= methodEndLine)
    .distinct
    .sorted

  vulnerableRanges.map { case (start, end) =>
    if (vertexLines.exists(line => line >= start && line <= end)) {
      (start, end)
    } else {
      val previous = vertexLines.filter(_ < start).lastOption.getOrElse(start)
      val following = vertexLines.find(_ > end).getOrElse(end)
      (previous, following)
    }
  }.distinct
}

def getVulnerableLines(method: Method, sourceLines: Option[List[String]] = None): List[(Int, Int)] = {
  val commentResults = getVulnerableLinesFromComments(method)

  if (commentResults.nonEmpty) {
    commentResults
  } else {
    sourceLines match {
      case Some(lines) => getVulnerableLinesFromSource(method, lines)
      case None => List.empty
    }
  }
}

def removeVulMarkers(input: String): String = {
  input
    .replaceAll("""/\*<vul-start>\*/""", "")
    .replaceAll("""/\*<vul-end>\*/""", "")
    .trim
}

def forwardSlice(
    node: StoredNode,
    edges: List[GraphSerializer.LineEdge],
    allVertices: List[StoredNode]
): List[StoredNode] = {
  val visited = scala.collection.mutable.Set[StoredNode]()
  val stack = scala.collection.mutable.Stack[StoredNode](node)

  while (stack.nonEmpty) {
    val current = stack.pop()
    if (!visited.contains(current)) {
      visited += current
      val outgoingEdges = edges.filter(
        _.srcLineNumber == current
          .property(PropertyNames.LineNumber)
          .asInstanceOf[Int]
      )
      val nextNodes = outgoingEdges.flatMap(edge =>
        allVertices.find(
          _.property(PropertyNames.LineNumber)
            .asInstanceOf[Int] == (edge.dstLineNumber)
        )
      )
      stack.pushAll(nextNodes)
    }
  }

  visited.toList
}

def backwardSlice(
    node: StoredNode,
    edges: List[GraphSerializer.LineEdge],
    allVertices: List[StoredNode]
): List[StoredNode] = {
  val visited = scala.collection.mutable.Set[StoredNode]()
  val stack = scala.collection.mutable.Stack[StoredNode](node)

  while (stack.nonEmpty) {
    val current = stack.pop()
    if (!visited.contains(current)) {
      visited += current
      val incomingEdges = edges.filter(
        _.dstLineNumber == current
          .property(PropertyNames.LineNumber)
          .asInstanceOf[Int]
      )
      val prevNodes = incomingEdges.flatMap(edge =>
        allVertices.find(
          _.property(PropertyNames.LineNumber)
            .asInstanceOf[Int] == (edge.srcLineNumber)
        )
      )
      stack.pushAll(prevNodes)
    }
  }

  visited.toList
}

class CfgPlusGenerator {
  val edgeType: String = EdgeTypes.CFG

  def generate(methodNode: Method): Graph = {
    val vertices          = methodNode.cfgNode.l ++ List(methodNode, methodNode.methodReturn) ++ methodNode.parameter.l
    val verticesToDisplay = vertices.filter(cfgNodeShouldBeDisplayed)

    def edgesToDisplay(srcNode: StoredNode, visited: List[StoredNode] = List()): List[Edge] = {
      if (visited.contains(srcNode)) {
        List()
      } else {
        val children             = expand(srcNode).filter(x => vertices.contains(x.dst))
        val (visible, invisible) = children.partition(x => cfgNodeShouldBeDisplayed(x.dst))
        visible.toList ++ invisible.toList.flatMap { n =>
          edgesToDisplay(n.dst, visited ++ List(srcNode)).map(y => Edge(srcNode, y.dst, edgeType = edgeType))
        }
      }
    }

    val edges = verticesToDisplay.flatMap { v =>
      edgesToDisplay(v)
    }.distinct

    val allIdsReferencedByEdges = edges.flatMap { edge =>
      Set(edge.src.id, edge.dst.id)
    }

    Graph(
      verticesToDisplay
        .filter(node => allIdsReferencedByEdges.contains(node.id)),
      edges
    )
  }

  protected def expand(v: StoredNode): Iterator[Edge] =
    v._cfgOut.map(node => Edge(v, node, edgeType = edgeType))

  private def isConditionInControlStructure(v: StoredNode): Boolean = v match {
    case id: Identifier => id.astParent.isControlStructure
    case _              => false
  }

  private def cfgNodeShouldBeDisplayed(v: StoredNode): Boolean =
    isConditionInControlStructure(v) ||
      !(v.isInstanceOf[Literal] ||
        v.isInstanceOf[Identifier] ||
        v.isInstanceOf[Block] ||
        v.isInstanceOf[MethodParameterIn])

}
