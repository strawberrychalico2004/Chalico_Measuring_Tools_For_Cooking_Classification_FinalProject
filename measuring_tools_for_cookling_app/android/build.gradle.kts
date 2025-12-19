import com.android.build.gradle.LibraryExtension

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// Ensure Android library subprojects (including plugins in the pub cache) have a namespace
// when the module's own build.gradle does not specify one. This avoids AGP errors
// for packages that haven't migrated to the 'namespace' DSL yet.

// (Namespace fallback for older Android library plugins removed during revert)

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
