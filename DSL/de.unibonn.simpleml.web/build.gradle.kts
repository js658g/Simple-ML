val javaVersion: Int by rootProject.extra
val xtextVersion: String by rootProject.extra

// Plugins -------------------------------------------------------------------------------------------------------------

plugins {
    java
    kotlin("jvm")
    war
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(javaVersion))
    }
}

// Dependencies --------------------------------------------------------------------------------------------------------

dependencies {
    implementation(project(":de.unibonn.simpleml"))
    implementation("org.emfjson:emfjson-jackson:1.2.0")

    api(project(":de.unibonn.simpleml.ide"))
    api("org.eclipse.xtext:org.eclipse.xtext.xbase.web:$xtextVersion")
    api("org.eclipse.xtext:org.eclipse.xtext.web.servlet:$xtextVersion")

    providedCompile("org.eclipse.jetty:jetty-annotations:9.4.22.v20191022")
    providedCompile("org.slf4j:slf4j-simple:1.7.36")
}

// Source sets ---------------------------------------------------------------------------------------------------------

sourceSets {
    main {
        java.srcDirs("src", "src-gen")
    }
}

// Tasks ---------------------------------------------------------------------------------------------------------------

tasks.register<JavaExec>("jettyRun") {
    group = "run"
    description = "Starts an example Jetty server with your language"

    dependsOn(sourceSets.main.get().runtimeClasspath)
    classpath = sourceSets.main.get().runtimeClasspath.filter { it.exists() }
    mainClass.set("de.unibonn.simpleml.web.ServerLauncher")
    standardInput = System.`in`
}

tasks.war {
    webAppDirectory.set(file("src"))
    webXml = file("src/web.xml")
}
