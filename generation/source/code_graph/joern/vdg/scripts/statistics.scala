// @Time: 2025.4.29 21:30
// @Author: Noah Frontzkowski
// @Desc: RUN > joern --script statistics.scala --param filepath=test.c --param saveprefix=tmp

//> using dep io.circe:circe-core_3:0.14.7
//> using dep io.circe:circe-generic_3:0.14.7
//> using dep io.circe:circe-parser_3:0.14.7

import io.shiftleft.codepropertygraph.generated.nodes.{Method, MethodParameterIn, Local, Comment, ControlStructure, Call}
import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.semanticcpg.language._
import scala.collection.mutable
import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets
import io.circe.Json
import io.circe.syntax._
import io.circe.generic.auto._

// constants

// https://en.cppreference.com/w/cpp/language/types
val cppPrimitives: Set[String] = Set(
  "bool",
  "char",
  "signed char",
  "unsigned char",
  "char16_t",
  "char32_t",
  "wchar_t",
  "short",
  "short int",
  "signed short",
  "signed short int",
  "unsigned short",
  "unsigned short int",
  "int",
  "signed",
  "signed int",
  "unsigned",
  "unsigned int",
  "long",
  "long int",
  "signed long",
  "signed long int",
  "unsigned long",
  "unsigned long int",
  "long long",
  "long long int",
  "signed long long",
  "signed long long int",
  "unsigned long long",
  "unsigned long long int",
  "float",
  "double",
  "long double",
  "void"
)

val booleanOperators = Set(
  Operators.equals,
  Operators.notEquals,
  Operators.lessThan,
  Operators.lessEqualsThan,
  Operators.greaterThan,
  Operators.greaterEqualsThan,
  Operators.logicalAnd,
  Operators.logicalOr,
  Operators.logicalNot
)

val bitwiseOperators = Set(
  Operators.and,
  Operators.or,
  Operators.xor,
  Operators.not,
  Operators.shiftLeft,
  Operators.arithmeticShiftRight,
  Operators.logicalShiftRight
)

val fieldAccessOperators = Set(
  Operators.fieldAccess,
  Operators.indirectFieldAccess
)

val JOERN_OPERATOR_PLACEHOLDER = "<operator>."

val isCppPrimitive: String => Boolean = typeName => cppPrimitives.exists(typeName.contains(_))


def collectNodes(astNodes: List[StoredNode]): Json = {
  val nodes = mutable.Map[String, mutable.Map[String, Int]]().withDefaultValue(mutable.Map().withDefaultValue(0))
  val booleanExpressionCount = mutable.Map[Int, Int]().withDefaultValue(0)
  val bitwiseExpressionCount = mutable.Map[Int, Int]().withDefaultValue(0)
  val regexPatterns = mutable.ListBuffer[String]()
  val externalTypes = mutable.Set[String]()

  // determines if a call is the root of a (complex) expression
  val topLevelOperators = (operators: Set[String], parentCheck: Call => Boolean) => astNodes.collect {
    case call: Call if operators.contains(call.name) && parentCheck(call) => call
  }

  val topLevelBooleanOperators = topLevelOperators(booleanOperators, call =>
    !call.astParent.isInstanceOf[Call] || !booleanOperators.contains(call.astParent.asInstanceOf[Call].name)
  )

  val topLevelBitwiseOperators = topLevelOperators(bitwiseOperators, call =>
    !call.astParent.isInstanceOf[Call] || !bitwiseOperators.contains(call.astParent.asInstanceOf[Call].name)
  )

  val getRegexFromCall: Call => Iterator[String] = call =>
    call.astMinusRoot.filter(child => child.isLiteral && child.asInstanceOf[Literal].typeFullName.matches(".*char.*")).map(_.code.
      stripPrefix("\""). // Joern's parser double-escapes
      stripSuffix("\"").
      replace("\\\\", "\\"))

  def updateCounts(call: Call, operatorType: String, operators: Set[String], topLevelOperators: List[Call], expressionCount: mutable.Map[Int, Int]): Unit = {
    val operatorMap = nodes.getOrElseUpdate(operatorType, mutable.Map().withDefaultValue(0))
    operatorMap(call.name.replace(JOERN_OPERATOR_PLACEHOLDER, "")) += 1

    if (topLevelOperators.contains(call)) {
      val childOperators = call.astMinusRoot.isCall.count(desc => operators.contains(desc.name))
      expressionCount(childOperators) += 1
    }
  }

  astNodes.foreach {
    case call: Call =>
      if (booleanOperators.contains(call.name)) {
        updateCounts(call, "BOOLEAN_OPERATOR", booleanOperators, topLevelBooleanOperators, booleanExpressionCount)
      } else if (bitwiseOperators.contains(call.name)) {
        updateCounts(call, "BITWISE_OPERATOR", bitwiseOperators, topLevelBitwiseOperators, bitwiseExpressionCount)
      } else if (fieldAccessOperators.contains(call.name)) {
        val fieldAccessMap = nodes.getOrElseUpdate("FIELD_ACCESS", mutable.Map().withDefaultValue(0))
        fieldAccessMap(call.name.replace(JOERN_OPERATOR_PLACEHOLDER, "")) += 1
      }
      // this is a bit lazy, but there is no perfect way to do this
      if (call.name.matches("(?i).*regex.*")) {
        getRegexFromCall(call).foreach(regexPatterns += _)
      }
    //case literal: Literal =>
    //      if (literal.typeFullName.matches(".*char.*")) {
    //      }
    case controlStructure: ControlStructure =>
      val controlMap = nodes.getOrElseUpdate("CONTROL_STRUCTURE", mutable.Map().withDefaultValue(0))
      controlMap(controlStructure.controlStructureType) += 1
    case methodParameterIn: MethodParameterIn =>
      if (!isCppPrimitive(methodParameterIn.typeFullName)) {
        externalTypes += methodParameterIn.typeFullName
      }
    case local: Local =>
      if (!isCppPrimitive(local.typeFullName)) {
        externalTypes += local.typeFullName
      }
    case _ =>
  }

  def expressionCountToJson(expressionCount: mutable.Map[Int, Int]): Json = {
    expressionCount.map { case (numChildren, count) =>
      Json.obj(
        "numChildren" -> Json.fromInt(numChildren),
        "count" -> Json.fromInt(count)
      )
    }.toList.asJson
  }

  val booleanExpressionJson = expressionCountToJson(booleanExpressionCount)
  val bitwiseExpressionJson = expressionCountToJson(bitwiseExpressionCount)

  nodes.map { case (k, v) => k -> v.toMap.asJson }.
    asJson.
    deepMerge(Json.obj(
      "BOOLEAN_EXPRESSION" -> booleanExpressionJson,
      "BITWISE_EXPRESSION" -> bitwiseExpressionJson,
      "REGULAR_EXPRESSION" -> Json.arr(regexPatterns.
        map(Json.fromString).toSeq: _*),
      "EXTERNAL_TYPE" -> Json.arr(externalTypes.
        map(Json.fromString).toSeq: _*)
    ))
}

// Scala is really neat actually
def countVulnerableTags(method: Method, vulStartComments: List[(Comment, Int)], vulEndComments: List[(Comment, Int)]): (Int, Int) = {
  val methodStartLine = method.lineNumber.map(_.toInt).getOrElse(Int.MaxValue)
  val methodEndLine = method.lineNumberEnd.map(_.toInt).getOrElse(Int.MinValue)

  val vulStartTags = vulStartComments.filter { case (_, line) => line >= methodStartLine && line <= methodEndLine }
  var vulEndTags = vulEndComments.filter { case (_, line) => line >= methodStartLine && line <= methodEndLine }

  vulStartTags.foldLeft((0, 0)) { case ((count, totalLines), (_, startLine)) =>
    val endTags = vulEndTags.filter { case (_, endLine) => endLine >= startLine }
    if (endTags.nonEmpty) {
      val (endTag, endLine) = endTags.minBy { case (_, endLine) => endLine } // closest end comment
      vulEndTags = vulEndTags.filterNot(_._1 == endTag)
      (count + 1, totalLines + (endLine - startLine + 1))
    } else {
      (count, totalLines)
    }
  }

}


def collectCpgStatistics(cpg: Cpg, statsBasePath: String): Unit = {
  val methods = cpg.method.filter(_.signature.nonEmpty).toList

  val vulStartComments = cpg.comment
    .filter(_.code.contains("vul-start"))
    .filter(_.lineNumber.isDefined)
    .map(c => (c, c.lineNumber.get.toInt))
    .toList

  var vulEndComments = cpg.comment
    .filter(_.code.contains("vul-end"))
    .filter(_.lineNumber.isDefined)
    .map(c => (c, c.lineNumber.get.toInt))
    .toList

  val methodStatisticsList = methods.map { method =>
    val methodStartLine = method.lineNumber.map(_.toInt).getOrElse(0)
    val methodEndLine = method.lineNumberEnd.map(_.toInt).getOrElse(0)
    val astNodes = method.ast.toList
    val nodes = collectNodes(astNodes)
    val numLines = (methodEndLine - methodStartLine) + 1
    val (vulnerableTags, vulnerableLines) = countVulnerableTags(method, vulStartComments, vulEndComments)
    Json.obj(
      "name" -> Json.fromString(method.name),
      "numLines" -> Json.fromInt(numLines),
      "vulTags" -> Json.fromInt(vulnerableTags),
      "vulLines" -> Json.fromInt(vulnerableLines),
      "nodes" -> nodes
    )
  }

  val json = Json.fromValues(methodStatisticsList)

  val outputPath = s"${statsBasePath}_stats.json"
  Files.write(Paths.get(outputPath), json.spaces2.getBytes(StandardCharsets.UTF_8))

  println(s"CPG statistics saved to $outputPath")
}

/**
 * This script collects node statistics and miscellaneous properties from C++ code segments.
 * It is specifically targeting the VulMaster dataset but is - in principle - language-agnostic.
 */
/*@main def exec(filepath: String, saveprefix: String) = {
  importCode.cpp(inputPath = filepath, args = List("--include-comments"))
  run.ossdataflow

  collectCpgStatistics(cpg, saveprefix)
}*/