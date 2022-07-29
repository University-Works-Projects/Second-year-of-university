<script>
export default {
  data() {
    return {
      titleClass: 'title'
    }
  }
}
</script>

<template>
  <!-- Those 3 following sintax produce the same result -->
  <h1 class='title'>Make me red</h1> <!-- CSS sintax -->
  <h1 v-bind:class='titleClass'>Make me red</h1> <!-- Vue sintax with v-bind -->
  <h1 :class='titleClass'>Make me red</h1> <!-- Vue syntax with the shortut for v-bind -->
</template>

<style>
.title {
  color: red;
}
</style>